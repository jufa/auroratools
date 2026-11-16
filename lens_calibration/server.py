"""Interactive alignment web application.

The page shows one frame with an *ideal* fisheye star overlay on top of it: a
zenith-pointing equiangular projection (180 deg across the disc) whose bright
catalogue stars are ringed, drawn to the measured size of the image circle.  An
alt-az grid underneath (rings at 30/60/90 deg zenith distance plus the cardinal
meridians) gives the eye something to line the sky up against, and the
brightest stars are named.  The user rotates, mirrors and drags the raster
underneath that overlay until the real stars sit in the rings, then presses
*Align* -- and :mod:`lens_calibration.match` takes over from the manual
alignment.

Dragging by eye is coarse, so the page can also be *georeferenced*: click a
star's circle and then the same star in the photograph, and ``/api/live``
refits the transform through every pair at once
(:meth:`lens_calibration.session.AlignmentSession.live_alignment`), returning
the pairs with the error each one still carries.  The page draws those as lines
and repeats until the user is done, so the manual step converges on the stars
the user recognises instead of on the disc's outline.

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

from .session import OVERLAY_MAG_LIMIT, WORK_HEIGHT, Alignment, AlignmentSession, _plain

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


class FrameStore:
    """The frame the page is editing, and the solve cached for it.

    The page can switch frames at any time, so nothing outside may hold on to a
    session: every endpoint takes the current one under the lock, and a new
    frame drops the cached solve along with the frame it belonged to.
    """

    def __init__(
        self,
        work_height: int = WORK_HEIGHT,
        calibration_dir: str | Path = "calibrations",
    ) -> None:
        self.work_height = int(work_height)
        self.calibration_dir = Path(calibration_dir)
        self.lock = threading.Lock()
        self.session: AlignmentSession | None = None
        self._key: tuple | None = None
        self._outcome = None

    def load(self, path: str | Path) -> AlignmentSession:
        """Read one frame and make it the current one.

        The frame is read *before* the lock is taken: decoding a 40 MB JPEG and
        fitting the image circle takes a moment, and holding the lock for it
        would stall every other request.  A frame that cannot be read leaves the
        old one in place.
        """
        session = AlignmentSession.open(
            path, work_height=self.work_height, calibration_dir=self.calibration_dir
        )
        with self.lock:
            self.session = session
            self.forget()
        return session

    def forget(self) -> None:
        """Drop the cached solve (the frame or the overlay changed)."""
        self._key = None
        self._outcome = None

    def outcome(
        self,
        session: AlignmentSession,
        alignment: Alignment,
        projection: str,
        fov_deg: float,
        mag_limit: float,
    ):
        """Solve from a manual alignment, once per distinct request."""
        key = (alignment, projection, float(fov_deg), float(mag_limit))
        if self._key != key:
            LOGGER.info("solving from manual alignment %s", alignment.to_dict())
            self._outcome = session.solve(
                alignment, projection=projection, fov_deg=fov_deg, mag_limit=mag_limit
            )
            self._key = key
        return self._outcome


def create_app(store: FrameStore) -> Flask:
    """Build the Flask application around one (replaceable) frame."""
    app = Flask(__name__)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_file(WEB_DIR / "index.html", mimetype="text/html", max_age=0)

    @app.get("/frame.jpg")
    def frame():
        session = store.session
        if session is None:
            return jsonify(NO_FRAME), 409
        return send_file(io.BytesIO(session.frame_jpeg), mimetype="image/jpeg", max_age=0)

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
        projection = args.get("projection") or "equidistant"
        fov_deg = _optional(args.get("fov"), float) or 180.0
        mag_limit = _optional(args.get("mag"), float)
        with store.lock:
            session = store.session
            if session is None:
                return jsonify(NO_FRAME), 409
            if utc is not None or latitude is not None or longitude is not None:
                session.configure(utc=utc, latitude=latitude, longitude=longitude)
                store.forget()
            alignment = Alignment.from_dict(
                {
                    "roll_deg": args.get("roll_deg", 0.0),
                    "mirrored": args.get("mirrored", "").lower() in ("1", "true", "yes"),
                    "dx": args.get("dx", 0.0),
                    "dy": args.get("dy", 0.0),
                }
            )
            options = {} if mag_limit is None else {"mag_limit": mag_limit}
            payload = session.summary(alignment, projection, fov_deg, **options)
        return jsonify(_plain(payload))

    @app.get("/api/detections")
    def api_detections():
        with store.lock:
            session = store.session
            if session is None:
                return jsonify(NO_FRAME), 409
            detections = session.detections.point_like()
            stars = [
                {"x": float(x), "y": float(y), "snr": float(snr), "flux": float(flux)}
                for x, y, snr, flux in zip(
                    detections.x, detections.y, detections.snr, detections.flux
                )
            ]
        return jsonify({"stars": stars, "count": len(stars)})

    @app.post("/api/align")
    def api_align():
        payload = request.get_json(silent=True) or {}
        alignment = Alignment.from_dict(payload)
        projection = str(payload.get("projection") or "equidistant")
        fov_deg = float(payload.get("fov_deg") or 180.0)
        mag_limit = float(payload.get("mag_limit") or 5.5)
        try:
            with store.lock:
                session = store.session
                if session is None:
                    return jsonify(NO_FRAME), 409
                outcome = store.outcome(session, alignment, projection, fov_deg, mag_limit)
                response = session.describe_solution(outcome, alignment, projection, fov_deg)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(_plain(response))

    @app.post("/api/live")
    def api_live():
        payload = request.get_json(silent=True) or {}
        alignment = Alignment.from_dict(payload)
        projection = str(payload.get("projection") or "equidistant")
        fov_deg = float(payload.get("fov_deg") or 180.0)
        mag_limit = float(payload.get("mag_limit") or OVERLAY_MAG_LIMIT)
        try:
            with store.lock:
                session = store.session
                if session is None:
                    return jsonify(NO_FRAME), 409
                response = session.live_alignment(
                    payload.get("points") or [], alignment, projection, fov_deg, mag_limit
                )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(_plain(response))

    @app.post("/api/save")
    def api_save():
        payload = request.get_json(silent=True) or {}
        alignment = Alignment.from_dict(payload)
        projection = str(payload.get("projection") or "equidistant")
        fov_deg = float(payload.get("fov_deg") or 180.0)
        mag_limit = float(payload.get("mag_limit") or 5.5)
        try:
            with store.lock:
                session = store.session
                if session is None:
                    return jsonify(NO_FRAME), 409
                outcome = store.outcome(session, alignment, projection, fov_deg, mag_limit)
                path = session.save_calibration(outcome, alignment, projection, fov_deg)
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


def _parse_utc(text: str) -> dt.datetime:
    """Parse an ISO-8601 instant, assuming UTC when no offset is given."""
    moment = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point (``python -m lens_calibration.server``)."""
    parser = argparse.ArgumentParser(
        description="Interactive fisheye alignment UI (manual alignment, then solve)"
    )
    parser.add_argument(
        "image", nargs="?",
        help="frame to open at start; omit it and pick one in the page",
    )
    parser.add_argument(
        "--work-height", type=int, default=WORK_HEIGHT,
        help=f"raster height used for display and solving (default {WORK_HEIGHT})",
    )
    parser.add_argument(
        "--calibration-dir", default="calibrations",
        help="where the save button writes JSON artefacts",
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
