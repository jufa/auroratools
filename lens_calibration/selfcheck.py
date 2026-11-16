"""Executable self-checks for the camera geometry and the attitude solver.

Run from the repository root::

    python -m lens_calibration.selfcheck

Why this exists
---------------
The synthetic frames used to test the solver are produced *through the same
projection the solver inverts*, so a handedness error cancels out of any single
round trip: pixels go out and come back with the same mistake baked in.  That
is exactly how a mirrored camera basis survived a green synthetic suite while
every real frame only matched with ``mirrored=True``.  The checks below
therefore pin down statements that a round trip cannot see:

1. the camera-to-ENU matrix is a proper rotation (``det == +1``) and the
   attitude round trips through its inverse;
2. a zenith-pointing camera with ``roll = 0`` really puts north at the top of
   the frame and *east on the left* -- looking up at the sky is not looking at
   it from above, and getting this backwards mirrors the whole star pattern;
3. stepping right and then down in the raster does not reflect the sky, and
   the ``mirrored`` flag really is a reflection.  This is the statement the
   old (y = image up) basis broke: the rotation matrix was proper, so item 1
   stayed green while every real frame could only match as a mirror;
4. pixels survive ``enu_from_pixel`` -> ``pixel_from_enu`` in both pixel
   conventions;
5. the public solve path recovers a known attitude from a synthetic frame
   without falling back on the mirrored convention;
6. the web page's own copy of the raster -> overlay transform, and the drag /
   rotate / mirror / nudge handlers that feed it, produce exactly what
   ``session.Alignment`` says they should -- the page is where a sign slip is
   most expensive, because the user sees it as "the overlay does not follow the
   photo" rather than as an error;
7. the page picks the frame it aligns: until one is open every endpoint says
   so, an unreadable upload is refused *without* replacing the frame already
   under the overlay, and both the picker and a drop put the chosen file on the
   wire.  The time/place panel *shows* what is in effect and is never sent back
   with a request: echoing it once served the frame just opened with the
   previous frame's clock and place, so its star circles were built for the old
   sky and appeared not to have been refreshed at all.  Endpoints under ``/api/``
   also answer *in JSON* when they do not exist or are called with the wrong
   method: Flask's HTML error page reached the page as "Unexpected token '<'",
   which reported a parse failure instead of a server that is out of date;
8. the star field can be built for a frame shot *today* without a network: the
   overlay and the solver both read it, so an IERS bulletin that has gone stale
   takes the whole page down (astropy raises rather than degrading quietly);
9. pinning a star and clicking it in the photograph is a georeferencing fit, not
   a nudge: one pin can only move the photo, two also set the roll, and a third
   makes the fit least squares -- so the page's residual line has to be exactly
   the reprojection error of the transform the page then applies, and the pixel
   it posts has to be a pixel of the *photograph*, not of the screen, or every
   refit of a turned photo would turn it again.  A wrong mirror is invisible with
   two pins (any rigid fit through two pairs is exact) and unmissable with three,
   which is why the mirror stays a user decision instead of being fitted;
10. from the *fourth* pin the pins also calibrate the lens.  A rigid fit is
    exhausted at two pins, so the third can only report an error it has no term
    for; what is missing is radial, and the model that describes it is the one
    ``cv2.fisheye`` calibrates (focal length plus odd ``k`` terms in theta).
    Four pins determine the optical centre and ``k1``, each further pin buys one
    more odd term, and the answer has to carry the overlay re-projected through
    that lens -- with the rings still remembering their *ideal* positions,
    because those are what a later pin's field angle is read from;
11. the drawn help is help and not data: the clusters, nebulae and Milky Way
    band that the page shows as orientation cues have to be placed by the sky
    model (the Pleiades marker on Alcyone's ring, a fixed pair of aids keeping
    its separation while the sky turns), they have to stay out of the ``stars``
    list the page's hit test walks -- that is what stops one from being pinned --
    and they have to follow the lens that is being drawn.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .catalog import StarField, star_field
from .geometry import (
    CameraModel,
    attitude_from_rotation,
    enu_from_altaz,
    rotation_from_attitude,
)
from .match import DetectedStars, solve_attitude
from .model import LensModel

#: Circular fisheye used by the calibration: 190 deg across the disc.
NOMINAL_FOV_DEG = 190.0


def nominal_lens() -> LensModel:
    """The equisolid first guess of the Meike 6.5 mm fisheye."""
    return LensModel.from_family("equisolid", fov_deg=NOMINAL_FOV_DEG)


def _camera(roll: float = 0.0, tilt_x: float = 0.0, tilt_y: float = 0.0) -> CameraModel:
    """Zenith-ish camera with the nominal lens (attitude in radians)."""
    return CameraModel(lens=nominal_lens(), roll=roll, tilt_x=tilt_x, tilt_y=tilt_y)


def check_rotation() -> None:
    """The attitude <-> matrix mapping is a proper rotation and round trips."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        roll = float(rng.uniform(-math.pi, math.pi))
        tilt_x = float(rng.uniform(-0.4, 0.4))
        tilt_y = float(rng.uniform(-0.4, 0.4))
        rotation = rotation_from_attitude(roll, tilt_x, tilt_y)

        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10), "not orthonormal"
        determinant = float(np.linalg.det(rotation))
        assert abs(determinant - 1.0) < 1e-10, (
            f"camera basis is improper (det={determinant:+.3f}): the frame is "
            "left handed, i.e. mirrored"
        )

        recovered = rotation_from_attitude(*attitude_from_rotation(rotation))
        assert np.allclose(rotation, recovered, atol=1e-9), (
            "attitude_from_rotation is not the inverse of rotation_from_attitude"
        )


def check_sky_orientation() -> None:
    """A zenith camera with ``roll = 0`` has north up and east on the left.

    A real photograph of the sky *from below*: north is at the top of the frame
    and a star rising in the east appears on the left-hand side.  The mirrored
    convention this module guards against put it on the right, which is what
    made every real frame match only as a reflection.
    """
    camera = _camera()
    height = 1000.0
    centre = 0.5 * height

    north_x, north_y = camera.pixel_from_altaz(math.radians(60.0), 0.0, height)
    east_x, east_y = camera.pixel_from_altaz(math.radians(60.0), 0.5 * math.pi, height)
    west_x, west_y = camera.pixel_from_altaz(math.radians(60.0), -0.5 * math.pi, height)

    assert abs(float(north_x) - centre) < 1e-6, "north is not on the vertical centre line"
    assert float(north_y) < centre, "north is not at the top of the frame"
    assert abs(float(east_y) - centre) < 1e-6 and abs(float(west_y) - centre) < 1e-6
    assert float(east_x) < centre, "east is on the right: the frame is mirrored"
    assert float(west_x) > centre, "west is on the left: the frame is mirrored"


def _orientation(model: CameraModel, height: float = 1000.0) -> float:
    """Sign carrier for ``(t_x x t_y) . d`` of the pixel -> sky map.

    ``t_x`` and ``t_y`` are the sky directions swept when stepping one pixel
    right and one pixel down.  The raster's own axes are ``x`` right and ``y``
    *down* with the scene ahead, so a real camera gives a positive triple
    product; a mirrored convention gives a negative one.
    """
    x, y, step = 520.0, 480.0, 1.0
    centre = model.enu_from_pixel(x, y, height)
    right = model.enu_from_pixel(x + step, y, height)
    down = model.enu_from_pixel(x, y + step, height)
    return float(np.dot(np.cross(right - centre, down - centre), centre))


def check_pixel_handedness() -> None:
    """The raster is not a mirror of the sky, whatever the attitude is.

    A rotation of the whole sky cannot change the sign, so this holds for every
    roll; flipping the pixel convention has to change it.  That is exactly the
    failure the old ``y = image up`` basis produced -- and because it comes from
    the raster-to-camera-vector map rather than from ``rotation_from_attitude``,
    a proper-rotation check alone does not see it.
    """
    camera = _camera()
    for roll in (0.0, 0.7, -2.4):
        model = camera.with_updates(roll=roll)
        assert _orientation(model) > 0.0, (
            "stepping right then down in the raster reflects the sky: the "
            "pixel -> sky map is mirrored"
        )
        assert _orientation(model.with_updates(mirrored=True)) < 0.0, (
            "the mirrored convention is not a mirror, so the solver's fallback "
            "would be pointless"
        )


def check_round_trip() -> None:
    """Pixels survive the pixel -> ENU -> pixel round trip in both conventions."""
    camera = _camera(roll=0.4, tilt_x=0.05, tilt_y=-0.03)
    height = 640.0
    rng = np.random.default_rng(1)
    radius = 0.8 * np.sqrt(rng.uniform(0.0, 1.0, 200))
    angle = rng.uniform(-math.pi, math.pi, 200)
    x = 0.5 * height + radius * np.cos(angle) * 0.5 * height
    y = 0.5 * height + radius * np.sin(angle) * 0.5 * height

    for mirrored in (False, True):
        model = camera.with_updates(mirrored=mirrored)
        back_x, back_y = model.pixel_from_enu(model.enu_from_pixel(x, y, height), height)
        assert np.allclose(back_x, x, atol=1e-6), f"x mismatch (mirrored={mirrored})"
        assert np.allclose(back_y, y, atol=1e-6), f"y mismatch (mirrored={mirrored})"


def _synthetic_field(rng: np.random.Generator, count: int) -> StarField:
    """A handful of bright stars spread over the sky, as a ``StarField``."""
    alt = rng.uniform(math.radians(20.0), math.radians(85.0), count)
    az = rng.uniform(0.0, 2.0 * math.pi, count)
    enu = np.stack(
        [np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1
    )
    return StarField(
        names=tuple(f"synthetic {index}" for index in range(count)),
        hip=np.arange(count),
        vmag=np.full(count, 2.0),
        bv=np.full(count, 0.6),
        alt=alt,
        az=az,
        enu=enu,
    )


def check_synthetic_solve() -> None:
    """The public solve path recovers a known attitude without mirroring."""
    rng = np.random.default_rng(2)
    lens = nominal_lens()
    truth = CameraModel(
        lens=lens,
        roll=math.radians(25.0),
        tilt_x=math.radians(2.0),
        tilt_y=math.radians(-3.0),
    )
    field = _synthetic_field(rng, 80)
    height = 800.0

    predicted_x, predicted_y = truth.pixel_from_altaz(field.alt, field.az, height)
    margin = 0.02 * height
    inside = (
        (predicted_x > margin)
        & (predicted_x < height - margin)
        & (predicted_y > margin)
        & (predicted_y < height - margin)
    )
    count = int(np.count_nonzero(inside))
    assert count >= 25, f"only {count} synthetic stars landed in the frame"

    detections = DetectedStars(
        x=predicted_x[inside],
        y=predicted_y[inside],
        flux=np.linspace(20.0, 1.0, count),
        peak=np.full(count, 5.0),
        fwhm=np.full(count, 1.5),
        snr=np.full(count, 40.0),
        height_px=height,
    )

    solution = solve_attitude(
        detections,
        field,
        _camera(),  # zenith pointing, roll 0: the solver has to find the rest
        height,
        height,
        tilt_range_deg=6.0,
        fit_geometry=False,
        calibrate_lens=False,
        quality_filter=False,
    )
    delta = solution.camera.rotation().T @ truth.rotation()
    error_deg = math.degrees(
        math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    )
    print(
        f"    solved {len(solution.correspondences)}/{count} stars, "
        f"{solution.correspondences.rms_deg:.3f} deg RMS, "
        f"{error_deg:.3f} deg attitude error"
    )
    for note in solution.notes:
        print(f"    note: {note}")

    assert not solution.mirrored, (
        "solve_attitude fell back to the mirrored pixel convention: the camera "
        "frame is left handed again"
    )
    assert solution.match_fraction > 0.5, (
        f"only {len(solution.correspondences)} of {count} synthetic stars matched"
    )
    assert error_deg < 1.0, f"attitude is {error_deg:.2f} deg away from the truth"


def check_web_ui() -> None:
    """The browser half of the alignment page agrees with the Python half.

    The page keeps its own copy of the raster -> overlay transform (it has to:
    the photo is moved by a CSS transform, and dragging must feel 1:1), so the
    copy is checked against :meth:`session.Alignment.forward` instead of being
    trusted.  The event handlers are fired too, because a sign slip in the drag
    or the mirror flag would only show up as "the overlay does not follow the
    photo" after ten minutes of dragging.

    The frame-picker path is driven the same way: the chosen file goes on the
    wire, and the request that follows carries no site, so the frame just
    opened is served with its *own* EXIF time and place -- the panel shows what
    is in effect, it does not impose it on the next frame.

    Skipped when node is unavailable; the page itself is always inspected for
    the structural mistakes that are easy to make when it is edited by hand.
    """
    from .session import Alignment

    web = Path(__file__).parent / "web"
    page_path = web / "index.html"
    text = page_path.read_text()
    assert text.count("<script>") == 1 and text.count("</script>") == 1, (
        "index.html must hold exactly one script block"
    )
    assert "__NEXT" not in text, "chunk marker left behind in index.html"
    for needle in ("function forward(", "function applyTransform(", "function siteQuery(",
                   'id="showGrid"', 'id="showAids"', 'id="openFile"', "/api/open",
                   "overlay.grid", "overlay.aids", "data-remove", "strokeText"):
        assert needle in text, f"index.html lost {needle!r}"

    node = shutil.which("node")
    if node is None:
        print("    node not found: skipping the browser-side transform check")
        return

    # One table of transforms from the authoritative implementation, so the JS
    # is compared against Python and not against a second copy of the maths.
    width = height = 1400.0
    aligns = [
        Alignment(),
        Alignment(roll_deg=33.4),
        Alignment(roll_deg=-119.3, mirrored=True, dx=25.5, dy=-14.25),
        Alignment(roll_deg=90.0, mirrored=True),
        Alignment(roll_deg=-137.0, dx=-61.0, dy=42.0),
    ]
    points = [(0.0, 0.0), (width, height), (0.5 * width, 0.5 * height),
              (431.7, 918.2), (12.0, 1391.5)]
    transforms = []
    for index, align in enumerate(aligns):
        for x, y in points:
            fx, fy = align.forward(x, y, width, height)
            transforms.append({
                "name": f"align{index} ({x:.1f}, {y:.1f})",
                "roll_deg": align.roll_deg,
                "mirrored": align.mirrored,
                "dx": align.dx,
                "dy": align.dy,
                "x": x,
                "y": y,
                "want_x": float(fx),
                "want_y": float(fy),
            })
    spec = {
        "image": {"width": width, "height": height},
        "stage_css": {"width": 700, "height": 700, "scale": 0.5},
        "transforms": transforms,
    }

    with tempfile.TemporaryDirectory() as tmp:
        cases = Path(tmp) / "cases.json"
        cases.write_text(json.dumps(spec))
        run = subprocess.run(
            [node, str(web / "check_page.js"), str(page_path), str(cases)],
            capture_output=True, text=True,
        )
    print("\n".join("    " + line for line in run.stdout.splitlines()))
    if run.stderr.strip():
        print("\n".join("    " + line for line in run.stderr.splitlines()[-5:]))
    assert run.returncode == 0 and "PAGE_CHECK_OK" in run.stdout, (
        "the page's transform or handlers disagree with session.Alignment"
    )


def check_frame_picker() -> None:
    """The page chooses the frame, and a frame that fails to open keeps the old.

    The UI used to be pinned to one image from the command line.  Now the frame
    arrives over ``/api/open``, so two statements have to hold: an endpoint that
    needs a frame answers 409 (not a traceback) until one is opened, and an
    upload the server cannot read is reported instead of quietly leaving the
    previous photo under the overlay -- the user would then align the wrong
    frame and never see a thing wrong.
    """
    from .server import UPLOAD_DIR, FrameStore, create_app

    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    store = FrameStore(calibration_dir=Path(tempfile.mkdtemp()))
    client = create_app(store).test_client()

    for url in ("/api/session", "/api/detections", "/frame.jpg"):
        assert client.get(url).status_code == 409, f"{url} answered before a frame was open"
    for url in ("/api/align", "/api/save"):
        assert client.post(url, json={}).status_code == 409, f"{url} answered without a frame"

    empty = client.post("/api/open", json={})
    assert empty.status_code == 400 and "path" in empty.get_json()["error"], empty.get_json()
    missing = client.post("/api/open", json={"path": "/nonexistent/frame.jpg"})
    assert missing.status_code == 404, missing.get_json()
    junk = client.post(
        "/api/open",
        data={"file": (io.BytesIO(b"not a jpeg"), "junk.jpg")},
        content_type="multipart/form-data",
    )
    assert junk.status_code == 400, junk.get_json()
    assert not (UPLOAD_DIR / "junk.jpg").exists(), "an unreadable upload was kept"
    assert store.session is None, "a failed open replaced the frame"

    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the open round trip")
        return
    try:
        opened = client.post("/api/open", json={"path": str(sample)})
        assert opened.status_code == 200, opened.get_json()
        assert opened.get_json()["name"] == sample.name, opened.get_json()
        assert Path(store.session.path).name == sample.name
        assert client.get("/api/session").status_code == 200
        assert client.get("/frame.jpg").mimetype == "image/jpeg"

        # the path the browser takes: the bytes are posted, not a file name
        posted = client.post(
            "/api/open",
            data={"file": (io.BytesIO(sample.read_bytes()), "posted.jpg")},
            content_type="multipart/form-data",
        )
        assert posted.status_code == 200, posted.get_json()
        assert Path(store.session.path).name == "posted.jpg", store.session.path
    finally:
        (UPLOAD_DIR / "posted.jpg").unlink(missing_ok=True)


def check_live_alignment() -> None:
    """Pinning stars is a fit, and the fit is what the page draws.

    Two statements have to hold for the live workflow to mean anything.  The
    first is arithmetic: one pin can only move the photo (the rotation is the
    user's), two pins also determine the roll, and a third pin turns the fit
    into a least-squares one -- so a pin that disagrees is *visible* instead of
    being absorbed.  The second is that the page's residual line is not a
    separate opinion: it must be exactly the reprojection error of the
    transform the page then applies, or the user would chase a number that
    describes a different photo.

    A wrong mirror is the one thing two pins cannot show (any rigid transform
    through two pairs is exact, mirrored or not), which is why the check pins
    the three-pin case where it does blow up -- and why the page keeps the
    mirror as a user decision rather than fitting it.
    """
    from .server import FrameStore, create_app
    from .session import Alignment, ControlPoint, fit_alignment

    width = height = 1400.0
    truth = Alignment(roll_deg=-119.3, mirrored=True, dx=25.5, dy=-14.25)
    circles = [(300.0, 400.0), (1100.0, 380.0), (900.0, 1200.0), (420.0, 980.0)]
    pairs = []
    for name, (star_x, star_y) in zip(("Vega", "Altair", "Deneb", "Polaris"), circles):
        x, y = truth.inverse(star_x, star_y, width, height)
        pairs.append(ControlPoint(name, star_x, star_y, float(x), float(y)))

    one, residual = fit_alignment(pairs[:1], width, height, roll_deg=12.0, mirrored=True)
    assert abs(one.roll_deg - 12.0) < 1e-9 and one.mirrored, (
        f"one pin rolled the photo by {one.roll_deg:.2f} deg: the rotation is the user's"
    )
    assert float(residual.max()) < 1e-9, residual

    two, residual = fit_alignment(pairs[:2], width, height, roll_deg=0.0, mirrored=True)
    assert abs(two.roll_deg - truth.roll_deg) < 1e-9, (two.roll_deg, truth.roll_deg)
    assert abs(two.dx - truth.dx) < 1e-9 and abs(two.dy - truth.dy) < 1e-9, two
    assert float(residual.max()) < 1e-9, residual

    # A pin that disagrees is the noise a least-squares fit cannot explain; the
    # pin carrying it has to keep it, not hand it to the honest ones.
    noisy = pairs[:2] + [
        ControlPoint(pairs[2].name, pairs[2].star_x, pairs[2].star_y, pairs[2].x + 40.0, pairs[2].y)
    ]
    _, residual = fit_alignment(noisy, width, height, roll_deg=0.0, mirrored=True)
    print(
        f"    three pins, one 40 px out: residuals "
        f"{', '.join(f'{value:.1f}' for value in residual)} px "
        f"({float(np.sqrt(np.mean(residual**2))):.1f} px RMS)"
    )
    assert residual[2] > 1.4 * float(residual[:2].max()), residual
    assert float(residual.min()) > 5.0, residual

    # Two pins fit any similarity exactly, so a wrong mirror hides; three do not.
    _, hidden = fit_alignment(pairs[:2], width, height, mirrored=False)
    _, caught = fit_alignment(pairs[:3], width, height, mirrored=False)
    assert float(hidden.max()) < 1e-9 and float(caught.min()) > 100.0, (hidden, caught)

    store = FrameStore(calibration_dir=Path(tempfile.mkdtemp()))
    client = create_app(store).test_client()
    assert client.post("/api/live", json={}).status_code == 409, (
        "pinning was answered before a frame was open"
    )

    # An endpoint the page asks for and this server does not have has to answer
    # in JSON as well.  It is HTML by default, and the page -- which reads every
    # reply as JSON -- then reported "Unexpected token '<'" in the corner of the
    # screen instead of naming what was wrong.  A page loading while an older
    # server still runs is exactly how that happens.
    missing = client.post("/api/live_typo", json={})
    assert missing.status_code == 404 and missing.is_json, missing.get_data(as_text=True)[:80]
    assert "no POST /api/live_typo" in missing.get_json()["error"], missing.get_json()
    wrong_method = client.get("/api/live")
    assert wrong_method.status_code == 405 and wrong_method.is_json, wrong_method.status_code
    assert "no GET /api/live" in wrong_method.get_json()["error"], wrong_method.get_json()

    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the endpoint round trip")
        return
    assert client.post("/api/open", json={"path": str(sample)}).status_code == 200
    session = store.session
    image_w, image_h = float(session.width), float(session.height)
    # Pin the circles the page actually shows.  Through the session's own
    # alignment the circles sit on the photo, so two of these clicks are exact
    # and the third -- 30 px off -- is the only error the fit has to explain.
    shown = client.get("/api/session").get_json()
    stars = shown["overlay"]["stars"]
    assert len(stars) >= 3, f"the overlay only found {len(stars)} bright stars to pin"
    seed = Alignment.from_dict(shown["alignment"])
    pins = []
    for index, star in enumerate(stars[:3]):
        x, y = seed.forward(star["x"], star["y"], image_w, image_h)
        if index == 2:
            x += 30.0
        pins.append(
            {
                "name": star["name"],
                "star_x": star["x"],
                "star_y": star["y"],
                "x": float(x),
                "y": float(y),
            }
        )

    # A refit with nothing to fit, or with a pair that is not a pair, is a
    # mistake on the wire rather than an answer of "no change".
    assert client.post("/api/live", json={"points": []}).status_code == 400
    assert client.post("/api/live", json={"points": [{"name": "Vega"}]}).status_code == 400

    first = client.post("/api/live", json={"points": pins})
    assert first.status_code == 200, first.get_json()
    body = first.get_json()
    assert len(body["points"]) == len(pins), body["points"]
    assert body["max_px"] > 10.0 and body["rms_px"] > 5.0, body

    # The line the page draws is the reprojection error of *this* transform.
    applied = Alignment.from_dict(body["alignment"])
    error = np.zeros((len(pins), 2))
    for index, point in enumerate(body["points"]):
        fx, fy = applied.forward(point["x"], point["y"], image_w, image_h)
        drawn = math.hypot(fx - point["star_x"], fy - point["star_y"])
        assert abs(drawn - point["residual_px"]) < 1e-6, (drawn, point["residual_px"])
        error[index] = (point["star_x"] - fx, point["star_y"] - fy)

    # And it is the *least-squares* answer: what the fit leaves behind carries no
    # net shift or turn, so the third pin is shared out over the triangle instead
    # of either being ignored or bending the photo onto itself.  (The photo is
    # not allowed to zoom -- the fit is rigid -- so the residual is free to hold
    # a scale error, which is precisely what the solver is for.)
    clicked = np.array([[point["x"], point["y"]] for point in body["points"]])
    fx, fy = applied.forward(clicked[:, 0], clicked[:, 1], image_w, image_h)
    spread = np.column_stack([fx, fy])
    spread = spread - spread.mean(axis=0)
    assert float(np.abs(error.sum(axis=0)).max()) < 1e-6, error.sum(axis=0)
    assert abs(float(np.sum(error[:, 0] * spread[:, 1] - error[:, 1] * spread[:, 0]))) < 1e-6, (
        "the fit still owes a turn"
    )

    # The pairs are the pixels the user clicked, so the answer depends on the
    # pairs and not on the transform the request happened to carry along: three
    # pins set the roll and the shift themselves.
    nudged = client.post(
        "/api/live",
        json={"points": pins, "roll_deg": 37.0, "dx": 120.0, "dy": -40.0},
    )
    assert nudged.status_code == 200, nudged.get_json()
    other = nudged.get_json()
    assert [p["x"] for p in other["points"]] == [p["x"] for p in body["points"]]
    assert abs(other["rms_px"] - body["rms_px"]) < 1e-6, (other["rms_px"], body["rms_px"])

    # The mirror is the opposite case: it is the user's statement about how the
    # photo sits under the sky, so it is taken from the request and three pins
    # then say loudly that it is wrong.
    flipped = client.post("/api/live", json={"points": pins, "mirrored": True})
    assert flipped.status_code == 200, flipped.get_json()
    assert flipped.get_json()["rms_px"] > 100.0, (
        "a mirrored fit through three pins should not agree with them"
    )
    print(
        f"    {len(pins)} pins fitted through /api/live: {body['rms_px']:.1f} px RMS, "
        f"{body['max_px']:.1f} px worst ({body['points'][2]['name']} clicked 30 px off)"
    )


def check_sky_aids() -> None:
    """The aids are landmarks for the eye and never become control points.

    Clusters, nebulae and the Milky Way are drawn to answer "is that smudge the
    Pleiades or the Double Cluster?", which is the one question the rings cannot
    answer while the photo is still being dragged.  Three statements have to
    hold, or the layer is decoration with a plausible look:

    1. they are placed by the *sky model*, not by hand: the Pleiades marker has
       to land on Alcyone's ring, and the separation between two aids has to be
       the same at two instants while their alt/az has rotated -- a projection
       slip would move them, and a hard-coded screen position would not rotate;
    2. they are a separate list from ``stars``, which is the list the page's hit
       test walks, so no aid can ever be picked as a pin;
    3. they follow the lens being drawn: the calibrated overlay carries the layer
       too, and the band is a band (a filled strip of samples, not a point), with
       every sample inside the disc.
    """
    from .aids import AID_KINDS, SKY_AIDS, band_half_width_deg, sky_aids
    from .geometry import CameraModel
    from .server import FrameStore, create_app

    # --- the table itself ---------------------------------------------------
    names = [aid.name for aid in SKY_AIDS]
    assert len(names) == len(set(names)), "duplicate aid names"
    for aid in SKY_AIDS:
        assert aid.kind in AID_KINDS, (aid.name, aid.kind)
        assert 0.0 < aid.radius_deg < 12.0, (aid.name, aid.radius_deg)
        assert -90.0 <= aid.dec_deg <= 90.0 and 0.0 <= aid.ra_deg < 360.0, aid
    assert len(SKY_AIDS) >= 20, len(SKY_AIDS)
    # Both hemispheres: a southern frame has to have landmarks of its own.
    assert min(aid.dec_deg for aid in SKY_AIDS) < -60.0, "no far-southern aid"
    assert max(aid.dec_deg for aid in SKY_AIDS) > 50.0, "no far-northern aid"

    # The band is widest at the bulge, narrow at the anticentre, has its second
    # bump at Cygnus, and wraps with galactic longitude.
    width = band_half_width_deg(np.array([0.0, 90.0, 180.0, 80.0, 100.0]))
    assert width[0] > width[1] > width[2], width          # widest at the bulge
    assert width[3] > width[4], width                     # Cygnus is a bump
    assert width.min() > 4.0 and width.max() < 20.0, width
    wrapped = float(band_half_width_deg(10.0)) - float(band_half_width_deg(370.0))
    assert abs(wrapped) < 1e-9, "the band is not periodic in galactic longitude"

    # --- placement comes from the sky --------------------------------------
    when = dt.datetime(2026, 1, 18, 9, 15, 17, tzinfo=dt.timezone.utc)
    first = sky_aids(when, 64.5724, -149.262)
    second = sky_aids(when + dt.timedelta(hours=3), 64.5724, -149.262)
    assert first.names == second.names, "the aid list follows the horizon"
    assert len(first) > 5, len(first)
    assert first.band_alt.shape[0] == 2 and first.band_alt.shape[1] > 100, (
        first.band_alt.shape
    )
    index = {name: position for position, name in enumerate(first.names)}
    pleiades, hyades = index["Pleiades (M45)"], index["Hyades"]

    def separation(sky, a: int, b: int) -> float:
        """Angle between two aids, measured from their alt/az and nothing else."""
        pair = enu_from_altaz(sky.alt[[a, b]], sky.az[[a, b]])
        return math.degrees(math.acos(float(np.clip(pair[0] @ pair[1], -1.0, 1.0))))

    moved_sky = abs(math.degrees(float(first.alt[pleiades] - second.alt[pleiades])))
    assert moved_sky > 5.0, f"three hours moved the Pleiades only {moved_sky:.2f} deg"
    # The pair is fixed on the sky, so its separation cannot change: a frame
    # error (a sign, a precession epoch) would move it by degrees, while the
    # transform itself is good to ~1e-5 deg.
    drift = abs(separation(first, pleiades, hyades) - separation(second, pleiades, hyades))
    assert drift < 1e-4, f"a fixed pair of aids changed separation by {drift:.2e} deg"

    # --- the overlay wires them in, beside the rings ------------------------
    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the overlay half of the aid check")
        return
    store = FrameStore(calibration_dir=Path(tempfile.mkdtemp()))
    client = create_app(store).test_client()
    assert client.post("/api/open", json={"path": str(sample)}).status_code == 200
    session = store.session
    shown = client.get("/api/session?projection=equidistant&fov=180&mag=4.5").get_json()
    overlay = shown["overlay"]
    aids = overlay["aids"]
    rings = {ring["name"]: ring for ring in overlay["stars"]}

    assert aids["objects"], "the overlay carries no sky aids at all"
    assert "vmag" not in aids["objects"][0], "an aid is dressed as a catalogue star"
    assert not {aid["name"] for aid in aids["objects"]} & set(rings), (
        "an aid is also a ring, so the page could pin it"
    )
    centre = overlay["centre"]
    furthest = max(
        math.hypot(aid["x"] - centre["x"], aid["y"] - centre["y"]) for aid in aids["objects"]
    )
    assert furthest <= overlay["radius_px"] + 1e-6, (furthest, overlay["radius_px"])

    # The Pleiades marker has to sit on Alcyone -- the brightest Pleiad, and the
    # nearest ring to the cluster's centre -- not merely somewhere in the frame.
    pleiad = next(aid for aid in aids["objects"] if aid["name"].startswith("Pleiades"))
    nearest = min(
        overlay["stars"],
        key=lambda ring: math.hypot(ring["x"] - pleiad["x"], ring["y"] - pleiad["y"]),
    )
    gap = math.hypot(nearest["x"] - pleiad["x"], nearest["y"] - pleiad["y"])
    assert nearest["name"] == "Alcyone" and gap < 3.0, (nearest["name"], gap)

    # The band is a strip of samples, all of them inside the disc, and long
    # enough to be a band rather than a stub.
    assert aids["band"], "no Milky Way band in the overlay"
    for polygon in aids["band"]:
        assert len(polygon) >= 6, len(polygon)
        span = max(p[0] for p in polygon) - min(p[0] for p in polygon)
        tall = max(p[1] for p in polygon) - min(p[1] for p in polygon)
        assert max(span, tall) > 0.2 * overlay["radius_px"], (span, tall)
        assert all(
            math.hypot(x - centre["x"], y - centre["y"]) <= overlay["radius_px"] + 1.0
            for x, y in polygon
        ), "a band sample is outside the disc"
    assert len(aids["band"][0]) > 20, "the band is a stub, not a band"

    # --- and they follow the lens being drawn ------------------------------
    fitted = session.overlay(
        "equidistant", 180.0, 4.5,
        camera=CameraModel(lens=session.nominal_lens(), xc=0.5, yc=0.5),
    )
    ideal_x = {aid["name"]: aid for aid in aids["objects"]}
    moved_px = max(
        math.hypot(aid["x"] - ideal_x[aid["name"]]["x"],
                   aid["y"] - ideal_x[aid["name"]]["y"])
        for aid in fitted["aids"]["objects"]
        if aid["name"] in ideal_x
    )
    assert moved_px > 1.0, f"the aids did not follow the lens ({moved_px:.2f} px)"
    print(
        f"    {len(aids['objects'])} aids + a {len(aids['band'][0])}-point band: Pleiades "
        f"{gap:.1f} px from Alcyone, {moved_px:.1f} px of movement through the nominal "
        f"lens; {len(first)} of {len(SKY_AIDS)} aids above the horizon at the sample time"
    )


def check_field_for_today() -> None:
    """A frame shot today yields a star field on a machine with no network.

    This is a regression guard, not a tautology: with ``auto_max_age`` left at
    its default astropy raises ``ValueError`` once the predictive part of the
    IERS table is more than a month old, and that single ``ValueError`` reached
    the page as "overlay unavailable" *and* turned Align into a 400.  Frames
    captured months or years after the bundled bulletin are the normal case, so
    the field has to work without a download.
    """
    from astropy.utils import iers

    assert iers.conf.auto_download is False, "the IERS table would be fetched at run time"
    assert iers.conf.auto_max_age is None, (
        "a stale IERS bulletin would fail every frame shot after the predictive window"
    )
    now = dt.datetime.now(dt.timezone.utc)
    field = star_field(now, 55.8146, -97.9248, min_alt_deg=5.0)
    assert len(field.names) > 1000, f"only {len(field.names)} stars above the horizon"
    # The overlay is drawn from the same field, so ask the far end of the
    # pipeline for its rings: the failure was visible only after the whole
    # session summary had been built.
    assert field.enu.shape == (len(field.names), 3), field.enu.shape


def check_pinned_lens() -> None:
    """Four pinned stars calibrate the lens; three cannot.

    A rigid fit -- what the pins do up to the third star -- has three degrees of
    freedom (shift and rotation, unit scale), so a second pin already exhausts it
    and later pins can only report an error with no term to remove it.  The
    radial model has to come in instead, and it is the model ``cv2.fisheye``
    calibrates: focal length plus the odd ``k`` terms of

        r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8).

    Four pins are the count at which that becomes solvable (the optical centre and
    ``k1``), and each further pin buys one more odd term.  This check pins down
    the statements the page depends on: the counting rule, that the two forms of
    the model describe the same curve, that a synthetic pair of (f, k) comes back
    exactly, that a *frozen* focal length cannot absorb a lens that is simply
    wider than the overlay assumes, and that the wire answer carries a
    re-projected overlay whose rings still remember their ideal positions (a pin
    is defined against those, so losing them would make the fourth pin
    unusable).
    """
    from .model import (
        LensModel,
        equiangular_pixels_per_radian,
        fisheye_radius,
        fisheye_theta,
    )
    from .server import FrameStore, create_app
    from .session import (
        MIN_PINS_FOR_LENS,
        Alignment,
        ControlPoint,
        fit_alignment,
        fit_lens_from_pins,
        pin_coefficient_count,
    )

    assert MIN_PINS_FOR_LENS == 4, MIN_PINS_FOR_LENS
    # Four and five pins buy k1 and then k2; past that the order is capped, so a
    # sixth star and beyond over-determine the same five unknowns.
    counts = [pin_coefficient_count(count) for count in (3, 4, 5, 6, 7, 8)]
    assert counts == [1, 1, 2, 2, 2, 2], counts

    # The two forms are the same family: the odd polynomial fitted to the
    # OpenCV curve reproduces it to a small fraction of a pixel.
    height = 1400.0
    focal, distortion = 520.0, (-0.012,)
    converted = LensModel.from_fisheye(focal, distortion, height_px=height, sigma_edge=0.988)
    radius = np.linspace(0.0, 0.988 * 0.5 * height, 200)
    wanted = fisheye_theta(radius, focal, distortion)
    drift = np.degrees(np.abs(converted.theta(radius / (0.5 * height)) - wanted)).max()
    assert drift < 0.02, f"the polynomial form misses the fisheye curve by {drift:.4f} deg"

    # A synthetic lens the pins can see exactly: the ideal overlay's radius gives
    # the field angle, the photograph's radius is the distorted one.
    ideal = LensModel.from_family("equidistant", fov_deg=180.0, sigma_edge=0.988)
    focal_ideal = equiangular_pixels_per_radian(ideal, height)
    centre = (0.53 * 1400.0, 0.46 * 1400.0)
    pins = []
    for theta_deg, az_deg in ((20.0, 10.0), (48.0, 100.0), (70.0, 200.0), (86.0, 290.0),
                              (35.0, 45.0), (62.0, 150.0)):
        theta = math.radians(theta_deg)
        direction = np.array([math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))])
        ideal_radius = theta * focal_ideal
        seen_radius = float(fisheye_radius(theta, focal, distortion))
        pins.append(
            ControlPoint(
                f"synthetic {theta_deg:.0f}@{az_deg:.0f}",
                float(700.0 + ideal_radius * direction[0]),
                float(700.0 + ideal_radius * direction[1]),
                float(centre[0] + seen_radius * direction[0]),
                float(centre[1] + seen_radius * direction[1]),
            )
        )

    four = fit_lens_from_pins(pins[:4], Alignment(), 1400.0, height, ideal)
    assert four.num_coefficients == 1, four.num_coefficients
    assert abs(four.focal_px - focal) < 0.5, (four.focal_px, focal)
    assert abs(four.coefficients[0] - distortion[0]) < 1e-4, four.coefficients
    assert four.rms_px < 0.05, four.rms_px
    assert abs(four.centre_px[0] - centre[0]) < 0.5 and abs(four.centre_px[1] - centre[1]) < 0.5, (
        four.centre_px
    )
    five = fit_lens_from_pins(pins[:5], Alignment(), 1400.0, height, ideal)
    assert five.num_coefficients == 2 and abs(five.coefficients[1]) < 1e-3, five.coefficients
    assert abs(five.focal_px - focal) < 0.5, five.focal_px

    # The sixth star does not buy a third term: it is another equation for the
    # same five unknowns, so it cannot invent a coefficient the lens does not
    # have -- and the readout says which way those extra pins were spent.
    six = fit_lens_from_pins(pins, Alignment(), 1400.0, height, ideal)
    assert six.num_coefficients == 2, six.num_coefficients
    assert abs(six.coefficients[1]) < 1e-3, six.coefficients
    assert abs(six.focal_px - focal) < 0.5, six.focal_px
    capped = [note for note in six.notes if "theta^5" in note]
    assert capped and "over-determine" in capped[0], six.notes

    # Pinning is a loop, and it must not integrate its own deltas: the page turns
    # each click back through the alignment that is on screen *when the click is
    # taken*, so a pin is the photo pixel under the star, not a correction to the
    # previous fit.  Re-measure the same five stars through three different
    # alignments -- what the loop actually does as the photo jumps under the
    # pointer -- and the pins have to come out identical; otherwise the fifth fit
    # would drift away from the fourth, pin after pin.
    for on_screen in (
        Alignment(),
        Alignment(roll_deg=3.0, dx=-40.0, dy=18.0),
        Alignment(roll_deg=-7.5, mirrored=True, dx=12.0, dy=-25.0),
    ):
        measured = []
        for point in pins[:5]:
            click_x, click_y = on_screen.forward(point.x, point.y, 1400.0, height)
            x, y = on_screen.inverse(float(click_x), float(click_y), 1400.0, height)
            assert abs(x - point.x) < 1e-9 and abs(y - point.y) < 1e-9, (x, y, point)
            measured.append(ControlPoint(point.name, point.star_x, point.star_y, x, y))
        again = fit_lens_from_pins(measured, Alignment(), 1400.0, height, ideal)
        assert abs(again.focal_px - five.focal_px) < 1e-6, (again.focal_px, five.focal_px)
        drift = max(abs(a - b) for a, b in zip(again.coefficients, five.coefficients))
        assert drift < 1e-6, (again.coefficients, five.coefficients)

    # A frozen focal length cannot absorb the overlay's own field of view being
    # off: the linear term of the model is then nailed to the circle's radius, and
    # k1 is left trying to describe a pure scale error.  This is why the focal
    # length is fitted.
    frozen = fit_lens_from_pins(
        pins[:4], Alignment(), 1400.0, height, ideal, optimize_focal=False
    )
    assert frozen.focal_px == focal_ideal, frozen.focal_px
    assert frozen.rms_px > 20.0 * max(four.rms_px, 1e-6), (frozen.rms_px, four.rms_px)
    print(
        f"    synthetic: 4 pins -> f {four.focal_px:.2f} k1 {four.coefficients[0]:.5f}"
        f" ({four.rms_px:.4f} px), frozen f -> {frozen.rms_px:.1f} px"
    )

    try:
        fit_lens_from_pins(pins[:3], Alignment(), 1400.0, height, ideal)
    except ValueError as error:
        assert "at least 4" in str(error), error
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("three pins were allowed to calibrate the lens")

    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the lens round trip")
        return
    store = FrameStore(calibration_dir=Path(tempfile.mkdtemp()))
    client = create_app(store).test_client()
    assert client.post("/api/open", json={"path": str(sample)}).status_code == 200
    session = store.session
    width_px, height_px = float(session.width), float(session.height)
    shown = client.get("/api/session?projection=equidistant&fov=180&mag=4.5").get_json()
    rings = shown["overlay"]["stars"]
    solved = client.post(
        "/api/align",
        json={"projection": "equidistant", "fov_deg": 180.0, "mag_limit": 5.5},
    )
    assert solved.status_code == 200, solved.get_json()
    camera = store._outcome.solution.camera
    # Where each ring's star really is in the photograph, so the pins are honest.
    seen = {}
    for ring in rings:
        index = session.field.names.index(ring["name"])
        px, py = camera.pixel_from_altaz(
            np.array([session.field.alt[index]]),
            np.array([session.field.az[index]]),
            height_px,
        )
        seen[ring["name"]] = (float(px[0]), float(py[0]))
    every = [
        ControlPoint(ring["name"], ring["x"], ring["y"], *seen[ring["name"]]) for ring in rings
    ]
    page, _ = fit_alignment(every, width_px, height_px, 0.0, camera.mirrored)
    # Pins have to span the field -- a lens fitted from one radius band is being
    # extrapolated everywhere else -- so take them rim inwards rather than in the
    # order the catalogue happens to list them.
    by_sigma = sorted(every, key=lambda point: camera.sigma_from_pixel(point.x, point.y, height_px))
    spread = np.linspace(0, len(by_sigma) - 1, min(6, len(by_sigma))).astype(int)
    ordered = [by_sigma[index] for index in spread[::-1]]

    def post(count: int) -> dict:
        body = {
            "projection": "equidistant",
            "fov_deg": 180.0,
            "mag_limit": 4.5,
            "roll_deg": page.roll_deg,
            "mirrored": page.mirrored,
            "dx": page.dx,
            "dy": page.dy,
            "points": [
                {
                    "name": point.name,
                    "star_x": point.star_x,
                    "star_y": point.star_y,
                    "x": point.x,
                    "y": point.y,
                }
                for point in ordered[:count]
            ],
        }
        reply = client.post("/api/live", json=body)
        assert reply.status_code == 200, reply.get_data(as_text=True)[:200]
        return reply.get_json()

    three = post(3)
    assert three["lens"] is None and not three.get("lens_error"), three.get("lens_error")
    assert session.pinned_lens is None, "three pins left a lens behind"

    four_pins = post(4)
    lens = four_pins["lens"]
    assert lens is not None, four_pins.get("lens_error")
    assert lens["num_coefficients"] == 1, lens["num_coefficients"]
    assert lens["pins"] == 4 and lens["coefficients"], lens
    assert 300.0 < lens["focal_px"] < 900.0, lens["focal_px"]
    assert four_pins["rms_px"] < 10.0, four_pins["rms_px"]
    overlay = four_pins["overlay"]
    assert overlay["calibrated"] is True, overlay["calibrated"]
    # The rings keep their ideal positions -- a pin is defined against those, and
    # they are what carries its field angle -- while being drawn at the calibrated
    # place.  Losing the former would make the fifth pin meaningless.
    ideal_positions = {(ring["name"], ring["x"], ring["y"]) for ring in rings}
    for ring in overlay["stars"]:
        assert (ring["name"], ring["x"], ring["y"]) in ideal_positions, ring
        assert "draw_x" in ring and "draw_y" in ring, ring
    applied = Alignment.from_dict(four_pins["alignment"])
    for point in four_pins["points"]:
        foot = applied.forward(point["x"], point["y"], width_px, height_px)
        drawn = math.hypot(point["draw_x"] - float(foot[0]), point["draw_y"] - float(foot[1]))
        assert abs(drawn - point["residual_px"]) < 0.1, (drawn, point["residual_px"])
    assert abs(overlay["centre"]["x"] - lens["centre_px"]["x"]) < 1e-6, overlay["centre"]
    assert session.pinned_lens is not None and session.seed_lens() is session.pinned_lens.lens, (
        "the solve would not start from the lens the pins measured"
    )

    five_pins = post(5)
    assert five_pins["lens"]["num_coefficients"] == 2, five_pins["lens"]["num_coefficients"]
    print(
        f"    sample frame: 4 pins -> f {lens['focal_px']:.0f} px/rad"
        f" k {[round(k, 4) for k in lens['coefficients']]}"
        f" ({four_pins['rms_px']:.2f} px); 5 pins ->"
        f" {five_pins['lens']['num_coefficients']} k terms ({five_pins['rms_px']:.2f} px);"
        f" rim {overlay['theta_edge_deg']:.2f} deg"
    )

    # Undoing a pin drops back below the threshold, so the lens goes with it --
    # otherwise the solver would keep starting from a lens no pins describe.
    back = post(3)
    assert back["lens"] is None and session.pinned_lens is None, back.get("lens_error")


#: Run in order; each one is a statement the rest of the package relies on.
CHECKS = (
    check_rotation,
    check_sky_orientation,
    check_pixel_handedness,
    check_round_trip,
    check_synthetic_solve,
    check_web_ui,
    check_frame_picker,
    check_live_alignment,
    check_pinned_lens,
    check_sky_aids,
    check_field_for_today,
)


def main() -> int:
    """Run every check; the return value is a shell exit status."""
    failures = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {check.__name__}: {error or 'assertion failed'}")
        else:
            print(f"ok   {check.__name__}")
    if failures:
        print(f"{failures} of {len(CHECKS)} checks failed")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
