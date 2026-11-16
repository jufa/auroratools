"""Interactive alignment web application.

The page shows one frame with a fisheye star overlay on top of it: the bright
catalogue stars ringed where the frame's *camera* puts them.  Until a star is
pinned that camera is the rig's seed lens -- a focal length in pixels per radian
plus its odd ``k`` terms -- pointing at the zenith for the frame's EXIF time and
place, with an alt-az grid underneath (rings at fixed zenith distance plus the
cardinal meridians) to give the eye something to line the sky up against.  The
user rotates, mirrors and drags the photograph underneath it until the real stars
sit in the rings.

The photograph is served as the camera wrote it: full resolution, no tone curve,
byte for byte when it is a JPEG.  A night frame is nearly black, but lifting it
is a *view*, so the page carries one gamma slider -- flat until asked for -- and
nothing this server does to the pixels can be mistaken for part of the
measurement.  The other view is ``/corrected.png``: the frame resampled onto an
ideal equiangular lens through the camera the pins measure, full resolution and
the photograph's own size, so the page can swap it under the same overlay and
the user can flash between the two.  It is drawn when the button is pressed and
never written to disk.

Dragging by eye is coarse, so the page can also be *georeferenced*: click a
star's circle and then the same star in the photograph, and ``/api/live`` refits
the camera through every pair at once
(:meth:`lens_calibration.session.AlignmentSession.live_alignment`), returning
the pairs with the error each one still carries and the overlay re-projected
through the camera it just solved.  One pin places the optical axis, two turn the
sky, three also settle the *tilt* of the axis -- which is what makes a frame that
does not point at the zenith an ordinary frame instead of a special case -- and
from the fourth the pairs measure the lens as well.  ``/api/save`` writes that
camera: lens, axis, attitude and mirror, in pixels of the photograph, in a JSON
file beside the frame it describes.

Because a pair is a star's name and a pixel of the photograph, none of that
depends on where the photo is dragged: turning the frame after it has been pinned
moves the view and not the calibration.

There is no automatic solver and no star detector: both were tried, both were
worse than the person looking at the frame, and a wrong answer from either cost
more than the click it saved.

Which frame is being aligned is the page's business, not the command line's:
*Open image…* (or dropping a file on the stage) posts it to ``/api/open`` and
the server swaps the session under a lock, so one server can work through a
whole sequence without a restart.  An image may still be given up front::

    python -m lens_calibration.server                               # start empty
    python -m lens_calibration.server sampleimages/00000687.jpg

The page is one static file (``lens_calibration/web/index.html``) talking to a
handful of JSON endpoints; there is no build step and no client-side library.
Its raster -> overlay transform is a second copy of
:meth:`lens_calibration.session.Alignment.forward`, so
:mod:`lens_calibration.selfcheck` runs the page's own maths and handlers
against the Python original (``lens_calibration/web/check_page.js``, skipped
when node is not installed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import tempfile
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .session import (
    CORRECTED_URL,
    MAX_COEFFICIENTS,
    OVERLAY_MAG_LIMIT,
    SEED_PROJECTION,
    WORK_HEIGHT,
    Alignment,
    AlignmentSession,
    _plain,
)

LOGGER = logging.getLogger(__name__)

#: Where the single-page UI lives (next to this module, so the package stays
#: relocatable and ``python -m lens_calibration.server`` works from anywhere).
WEB_DIR = Path(__file__).resolve().parent / "web"

#: Frames posted by the page land here before they are read.  The session needs
#: a real file (the raster is read twice, and the calibration artefact is named
#: after it), so the upload keeps its own base name instead of a temp name.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "lens_calibration_uploads"

#: Sent by every endpoint that needs a frame when the page has not opened one.
NO_FRAME = {"error": "no frame is open -- choose an image first"}

#: Where a save goes when the frame is a browser upload.  A posted file is the
#: server's own copy under :data:`UPLOAD_DIR`, so the folder it was uploaded from
#: is not something to write to -- the browser does not say what it was, and the
#: upload is temporary by definition.
UPLOAD_CALIBRATION_DIR = Path("calibrations")


class FrameStore:
    """The frame the page is editing.

    The page can switch frames at any time, so nothing outside may hold on to a
    session: every endpoint takes the current one under the lock, and a new frame
    replaces the old one along with everything the pins had measured for it.
    """

    def __init__(
        self,
        work_height: int = WORK_HEIGHT,
        calibration_dir: str | Path | None = None,
    ) -> None:
        self.work_height = int(work_height)
        #: ``None`` -- the default -- saves each artefact *beside its photograph*;
        #: a directory overrides that (``--calibration-dir``), and is also where a
        #: browser upload's JSON goes.
        self.calibration_dir = None if calibration_dir is None else Path(calibration_dir)
        self.lock = threading.Lock()
        self.session: AlignmentSession | None = None

    def load(self, path: str | Path) -> AlignmentSession:
        """Read one frame and make it the current one.

        The frame is read *before* the lock is taken: decoding a 40 MB JPEG and
        fitting the image circle takes a moment, and holding the lock for it
        would stall every other request.  A frame that cannot be read leaves the
        old one in place.
        """
        target = Path(path)
        directory = self.calibration_dir
        if directory is None and target.parent == UPLOAD_DIR:
            # A browser upload is the server's own copy under the temp directory:
            # writing the artefact there would lose it, so it goes to the folder
            # saves have always gone to.
            directory = UPLOAD_CALIBRATION_DIR
        session = AlignmentSession.open(
            target, work_height=self.work_height, calibration_dir=directory
        )
        with self.lock:
            self.session = session
        return session


def create_app(store: FrameStore) -> Flask:
    """Build the Flask application around one (replaceable) frame."""
    app = Flask(__name__)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_file(WEB_DIR / "index.html", mimetype="text/html", max_age=0)

    @app.get("/frame.jpg")
    def frame():
        """The photograph itself: full resolution, no tone curve.

        A JPEG that carries no EXIF orientation is the file's own bytes, so the
        page draws exactly what the camera wrote; anything else arrives as a
        lossless PNG of the decoded array.  The work raster is not a display
        limit -- it is what the pins and the overlay are measured in -- and the
        stretch a night frame needs belongs to the page's *view* controls.
        """
        session = store.session
        if session is None:
            return jsonify(NO_FRAME), 409
        return send_file(
            io.BytesIO(session.frame_image),
            mimetype=session.frame_mimetype,
            max_age=0,
        )

    # ------------------------------------------------------------- open a frame
    @app.post("/api/open")
    def api_open():
        """Take over a new frame: a browser upload, or a path on this machine."""
        upload = request.files.get("file")
        posted = False
        if upload is not None and upload.filename:
            name = Path(upload.filename).name
            if name in ("", ".", ".."):
                return jsonify({"error": f"unusable file name: {upload.filename!r}"}), 400
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            target = UPLOAD_DIR / name
            upload.save(target)
            posted = True
            LOGGER.info("received %s (%d bytes)", name, target.stat().st_size)
        else:
            payload = request.get_json(silent=True) or {}
            text = str(payload.get("path") or "").strip()
            if not text:
                return jsonify({"error": 'send a "file" upload, or {"path": "..."}'}), 400
            target = Path(text).expanduser()
            if not target.is_file():
                return jsonify({"error": f"no such file: {target}"}), 404
        try:
            session = store.load(target)
        except (OSError, ValueError) as error:
            if posted:
                target.unlink(missing_ok=True)   # do not keep bytes we cannot read
            return jsonify({"error": f"cannot open {Path(target).name}: {error}"}), 400
        return jsonify(
            {
                "path": session.path,
                "name": Path(session.path).name,
                "width": int(session.width),
                "height": int(session.height),
            }
        )

    # ------------------------------------------------------------- api
    @app.errorhandler(404)
    @app.errorhandler(405)
    def api_missing(error):
        """Answer an unknown or wrongly-called endpoint in JSON.

        The page talks to ``/api/*`` and nothing else, so an HTML error page
        there is a bug of our own making: it reached the page as "Unexpected
        token '<'" instead of naming the endpoint that was missing.  A server
        running an older build -- the usual cause -- then at least says so.
        """
        if not request.path.startswith("/api/"):
            return error.get_response()
        method = request.method
        return jsonify({"error": f"no {method} {request.path} on this server"}), error.code

    @app.get("/api/session")
    def api_session():
        args = request.args
        utc = _optional(args.get("utc"), _parse_utc)
        latitude = _optional(args.get("lat"), float)
        longitude = _optional(args.get("lon"), float)
        projection = args.get("projection") or SEED_PROJECTION
        fov_deg = _optional(args.get("fov"), float) or 180.0
        mag_limit = _optional(args.get("mag"), float)
        corrected = _flag(args.get("corrected"))
        try:
            focal_px = _optional(args.get("focal"), float)
            coefficients = _coefficients(args.get("k"))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        with store.lock:
            session = store.session
            if session is None:
                return jsonify(NO_FRAME), 409
            if utc is not None or latitude is not None or longitude is not None:
                session.configure(utc=utc, latitude=latitude, longitude=longitude)
            alignment = _alignment(args)
            options = {} if mag_limit is None else {"mag_limit": mag_limit}
            try:
                payload = session.summary(
                    alignment, projection, fov_deg, focal_px=focal_px,
                    coefficients=coefficients, corrected=corrected, **options,
                )
            except ValueError as error:
                return jsonify({"error": str(error)}), 400
        return jsonify(_plain(payload))

    @app.get(CORRECTED_URL)
    def corrected():
        """The frame resampled onto an ideal equiangular lens, full resolution.

        Drawn when the page asks for it -- the button, not the page load --
        because it is a full-resolution resample and a PNG to match; the session
        caches the bytes against the camera they were drawn from, so pressing the
        button again with nothing changed costs a memory copy.  The store's lock
        is only taken to read the session: the resample runs without it, so a
        long render cannot stall the endpoints that are cheap.
        """
        args = request.args
        projection = args.get("projection") or SEED_PROJECTION
        fov_deg = _optional(args.get("fov"), float) or 180.0
        try:
            focal_px = _optional(args.get("focal"), float)
            coefficients = _coefficients(args.get("k"))
            alignment = _alignment(args)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        session = store.session
        if session is None:
            return jsonify(NO_FRAME), 409
        try:
            data = session.corrected_png(
                alignment, projection, fov_deg, focal_px, coefficients
            )
        except (OSError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return send_file(io.BytesIO(data), mimetype="image/png", max_age=0)

    @app.post("/api/live")
    def api_live():
        payload = request.get_json(silent=True) or {}
        alignment = Alignment.from_dict(payload)
        projection = str(payload.get("projection") or SEED_PROJECTION)
        fov_deg = float(payload.get("fov_deg") or 180.0)
        mag_limit = float(payload.get("mag_limit") or OVERLAY_MAG_LIMIT)
        try:
            focal_px = _optional(payload.get("focal_px"), float)
            coefficients = _coefficients(payload.get("coefficients"))
            with store.lock:
                session = store.session
                if session is None:
                    return jsonify(NO_FRAME), 409
                response = session.live_alignment(
                    payload.get("points") or [],
                    alignment,
                    projection,
                    fov_deg,
                    mag_limit,
                    focal_px=focal_px,
                    coefficients=coefficients,
                )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(_plain(response))

    @app.post("/api/save")
    def api_save():
        payload = request.get_json(silent=True) or {}
        alignment = Alignment.from_dict(payload)
        projection = str(payload.get("projection") or SEED_PROJECTION)
        fov_deg = float(payload.get("fov_deg") or 180.0)
        try:
            focal_px = _optional(payload.get("focal_px"), float)
            coefficients = _coefficients(payload.get("coefficients"))
            with store.lock:
                session = store.session
                if session is None:
                    return jsonify(NO_FRAME), 409
                if session.pins is None or not session.pins.measured:
                    return jsonify(
                        {
                            "error": (
                                "no lens measured yet: pin at least four stars, or the"
                                " file would hold the seed and not a calibration"
                            )
                        }
                    ), 400
                path = session.save_calibration(
                    alignment, projection, fov_deg, focal_px, coefficients
                )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"path": str(path), "bytes": path.stat().st_size})

    return app


# -------------------------------------------------------------------- helpers
def _optional(text, convert):
    """Convert a query parameter when it is present and non-empty."""
    if text is None or not str(text).strip():
        return None
    return convert(str(text))


def _flag(text) -> bool:
    """A checkbox-shaped parameter: ``1``, ``true`` or ``yes``, anything else no."""
    return str(text or "").strip().lower() in ("1", "true", "yes")


def _alignment(args) -> Alignment:
    """The display transform a query string carries: roll, mirror, shift.

    One spelling, because two endpoints now read it (``/api/session`` and
    ``/corrected.png``) and they have to agree: the corrected render and the
    overlay drawn over it are only a flash comparison if both were built from the
    same transform.
    """
    return Alignment.from_dict(
        {
            "roll_deg": args.get("roll_deg", 0.0),
            "mirrored": _flag(args.get("mirrored")),
            "dx": args.get("dx", 0.0),
            "dy": args.get("dy", 0.0),
        }
    )


def _coefficients(value) -> tuple[float, ...] | None:
    """The ``k`` terms of a fisheye, from a JSON list or a separated string.

    The page offers one text field (``-0.103, 0.005``) and posts what it parsed,
    so both spellings arrive here.  ``None`` means "whatever the seed is".
    """
    if value is None:
        return None
    try:
        items = (
            [part for part in str(value).replace(",", " ").split() if part]
            if isinstance(value, str)
            else list(value)
        )
    except TypeError:
        raise ValueError(f"the k terms must be a list or a string, got {value!r}") from None
    if not items:
        return None
    try:
        values = tuple(float(item) for item in items)
    except (TypeError, ValueError):
        raise ValueError(f"the k terms must be numbers, got {value!r}") from None
    if len(values) > MAX_COEFFICIENTS:
        raise ValueError(f"at most {MAX_COEFFICIENTS} k terms, got {len(values)}")
    return values


def _parse_utc(text: str) -> dt.datetime:
    """Parse an ISO-8601 instant, assuming UTC when no offset is given."""
    moment = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point (``python -m lens_calibration.server``)."""
    parser = argparse.ArgumentParser(
        description="Interactive fisheye alignment UI (drag, then pin stars)"
    )
    parser.add_argument(
        "image", nargs="?",
        help="frame to open at start; omit it and pick one in the page",
    )
    parser.add_argument(
        "--work-height", type=int, default=WORK_HEIGHT,
        help=(
            f"height of the raster the geometry is done in (default {WORK_HEIGHT});"
            " the page is served the photograph at its own size regardless"
        ),
    )
    parser.add_argument(
        "--calibration-dir", default=None,
        help=(
            "folder for the saved JSON artefacts; omit it and each one is written"
            " beside its photograph, which is where it belongs (a browser upload"
            " has no folder of its own, so it goes to ./calibrations)"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    store = FrameStore(work_height=args.work_height, calibration_dir=args.calibration_dir)
    if args.image:
        store.load(args.image)
    app = create_app(store)
    url = f"http://{args.host}:{args.port}/"
    LOGGER.info("serving %s at %s", args.image or "the frame picker", url)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
