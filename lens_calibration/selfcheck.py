"""Executable self-checks for the camera geometry, the pin fit and the page.

Run from the repository root::

    python -m lens_calibration.selfcheck

Why this exists
---------------
The synthetic frames used by these checks are produced *through the same
projection the code inverts*, so a handedness error cancels out of any single
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
5. the overlay is drawn with the rig's own lens -- a focal length in pixels
   per radian and its odd ``k`` terms -- rather than with an ideal disc, so the
   rings start out near the stars and a pinned star's field angle is read
   through the curve that drew its circle;
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
8. the star field can be built for a frame shot *today* without a network:
   the overlay reads it, so an IERS bulletin that has gone stale takes the
   whole page down (astropy raises rather than degrading quietly);
9. pinning a star and clicking it in the photograph is a camera fit, not a nudge.
   A pair is a star's *name* and a pixel of the photograph, so the sky direction
   comes from the catalogue and the answer cannot depend on where the photo is
   dragged -- which is what lets a frame be turned after it has been pinned.  The
   rungs grow one degree of freedom at a time: one pin places the optical axis,
   two add the roll, three add the *tilt* of the axis (an off-zenith frame, which
   the old shift-and-rotate alignment could never express -- a tilt stretches one
   side of the image and compresses the other), and three is also where a wrong
   mirror becomes decidable.  The page's residual line has to be exactly the
   reprojection error of the positions the answer carries;
10. from the *fourth* pin the pins also calibrate the lens, together with the axis
    and the attitude: the model ``cv2.fisheye`` calibrates (focal length plus odd
    ``k`` terms in theta), four pins determining the optical centre and ``k1`` and
    each further pin buying one more odd term up to ``k4``.  A synthetic (f, k) on
    a *tilted* camera has to come back exactly -- lens, axis, roll and tilt -- and
    a curve that folds inside the disc is reported rather than drawn as if the rim
    were measured;
11. the drawn help is help and not data: the clusters, nebulae and Milky Way
    band that the page shows as orientation cues have to be placed by the sky
    model (the Pleiades marker on Alcyone's ring, a fixed pair of aids keeping
    its separation while the sky turns), they have to stay out of the ``stars``
    list the page's hit test walks -- that is what stops one from being pinned --
    and they have to follow the lens that is being drawn;
12. the artefact is written *beside the photograph it describes* (a folder of
    JSONs keyed by file name is a second place to keep in step with the frames),
    except for a browser upload, whose copy only exists under the temp directory;
13. the corrected preview really is the *ideal* equiangular lens and not a second
    rendering of the measured one: a synthetic star lands where the equiangular
    curve puts its field angle, on the same azimuth; correcting a camera that is
    already ideal is the identity; the raster keeps the photograph's own size; and
    the overlay asked for with ``corrected=1`` is drawn through that same lens --
    drawn through the measured one it would show the correction itself as an
    error, which is the feature inverted.
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

import cv2
import numpy as np

from .catalog import StarField, star_field
from .geometry import (
    CameraModel,
    attitude_from_rotation,
    enu_from_altaz,
    rotation_from_attitude,
)
from .model import LensModel
from .session import WORK_HEIGHT

#: Stand-in lens for the geometry checks (a 190 deg equisolid disc).
NOMINAL_FOV_DEG = 190.0


def stand_in_lens() -> LensModel:
    """A plausible fisheye for checks that need a lens and not a calibration."""
    return LensModel.from_family("equisolid", fov_deg=NOMINAL_FOV_DEG)


def _camera(roll: float = 0.0, tilt_x: float = 0.0, tilt_y: float = 0.0) -> CameraModel:
    """Zenith-ish camera with the stand-in lens (attitude in radians)."""
    return CameraModel(lens=stand_in_lens(), roll=roll, tilt_x=tilt_x, tilt_y=tilt_y)


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
            "the mirrored convention is not a mirror, so the fitted camera's "
            "fallback would be pointless"
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
                   "overlay.grid", "overlay.aids", "data-remove", "strokeText",
                   'id="focal"', 'id="kTerms"', 'id="projection"', "fisheye",
                   "/corrected.png", 'id="corrected"', "setCorrected",
                   "[hidden] { display: none !important; }"):
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

    for url in ("/api/session", "/frame.jpg"):
        assert client.get(url).status_code == 409, f"{url} answered before a frame was open"
    for url in ("/api/save", "/api/live"):
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

        # What the page is given is the photograph, not a rendering of it: the
        # file's own bytes at the file's own size, so the browser can zoom into
        # what the sensor recorded and no stretch of the server's is baked in.
        served = client.get("/frame.jpg")
        assert served.mimetype == "image/jpeg"
        assert served.data == sample.read_bytes(), (
            "the frame served to the page is not the file as it lies on disk"
        )
        photo = cv2.imread(str(sample), cv2.IMREAD_COLOR)
        raster = (int(store.session.width), int(store.session.height))
        assert store.session.frame_size == (photo.shape[1], photo.shape[0]), (
            f"{store.session.frame_size} is not the photograph's own size"
        )
        assert raster[1] == WORK_HEIGHT and raster != store.session.frame_size, (
            "the work raster is not a downscale of the photograph"
        )
        shown = client.get("/api/session").get_json()["image"]
        assert (shown["frame_width"], shown["frame_height"]) == store.session.frame_size
        assert (shown["width"], shown["height"]) == raster

        # the path the browser takes: the bytes are posted, not a file name
        posted = client.post(
            "/api/open",
            data={"file": (io.BytesIO(sample.read_bytes()), "posted.jpg")},
            content_type="multipart/form-data",
        )
        assert posted.status_code == 200, posted.get_json()
        assert Path(store.session.path).name == "posted.jpg", store.session.path

        # A frame that is not a JPEG arrives as a lossless PNG of the array the
        # geometry was fitted to -- the same pixels the pins are measured on --
        # and an EXIF orientation (which the raster ignores and the browser would
        # honour) takes the same path.
        png = sample.parent / "00000003.png"
        if png.is_file():
            assert client.post("/api/open", json={"path": str(png)}).status_code == 200
            lossless = client.get("/frame.jpg")
            assert lossless.mimetype == "image/png", lossless.mimetype
            decoded = cv2.imdecode(np.frombuffer(lossless.data, np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None and np.array_equal(
                decoded, cv2.imread(str(png), cv2.IMREAD_COLOR)
            ), "the PNG served for a non-JPEG frame is not the raster the fit used"
    finally:
        (UPLOAD_DIR / "posted.jpg").unlink(missing_ok=True)


def check_live_alignment() -> None:
    """Pinning is a camera fit, and it grows one degree of freedom at a time.

    The old page moved the photograph under a fixed sky chart, which is a
    similarity with unit scale: three degrees of freedom, exhausted by the second
    pin, and a *tilted* frame -- one that does not look at the zenith -- cannot be
    expressed that way at all.  What the pins now measure is the camera itself, and
    this check pins down each rung of the ladder against a known answer:

    * one pin places the optical axis and leaves the attitude alone;
    * two pins add the roll, exactly;
    * three pins add the tilt of the axis, which is what makes an off-zenith frame
      an ordinary frame instead of a special case: a tilt stretches one side of
      the image and compresses the other, and no shift or rotation of the raster
      can imitate that;
    * a pin that disagrees keeps its error instead of handing it to the rest.

    Two statements are about the wire.  A pair is a star's *name* and a pixel of
    the *photograph*, so the answer cannot depend on where the photo happens to be
    dragged -- which is what lets a frame be turned after it has been pinned --
    and the line the page draws is the reprojection error of the positions the
    answer carries, so the two can never disagree.
    """
    from .geometry import CameraModel, altaz_from_enu, enu_from_altaz
    from .model import LensModel
    from .server import FrameStore, create_app
    from .session import (
        MIN_PINS_FOR_ATTITUDE,
        MIN_PINS_FOR_LENS,
        MIN_PINS_FOR_MIRROR,
        Alignment,
        ControlPoint,
        fit_camera_from_pins,
    )

    width = height = 1400.0
    assert (MIN_PINS_FOR_ATTITUDE, MIN_PINS_FOR_MIRROR, MIN_PINS_FOR_LENS) == (2, 3, 4), (
        MIN_PINS_FOR_ATTITUDE, MIN_PINS_FOR_MIRROR, MIN_PINS_FOR_LENS
    )
    lens = LensModel.from_fisheye(560.0, (-0.09, 0.004), height_px=height, sigma_edge=0.988)
    # Eight directions spread over the disc, as a photograph would offer them.
    directions = enu_from_altaz(
        np.radians(90.0 - np.array([84.0, 65.0, 40.0, 10.0, 84.0, 65.0, 40.0, 10.0])),
        np.radians(np.array([15.0, 195.0, 15.0, 195.0, 150.0, 30.0, 250.0, 100.0])),
    )
    flat = CameraModel(lens=lens, xc=0.523, yc=0.472, roll=math.radians(23.0))
    # An off-zenith frame: the optical axis 27 deg away from the zenith, which is
    # the thing a shift-and-rotate alignment could never express.
    tilted = flat.with_updates(tilt_x=0.42, tilt_y=-0.31)
    # Where a drag leaves the camera before any star is pinned: the zenith, with
    # the axis in the middle of the frame.
    start = CameraModel(lens=lens)

    def pins(camera, count):
        x, y = camera.pixel_from_enu(directions[:count], height)
        return [
            ControlPoint(f"star {index}", float(px), float(py))
            for index, (px, py) in enumerate(zip(x, y))
        ]

    def fit(camera, count, **options):
        return fit_camera_from_pins(
            pins(camera, count), directions[:count], width, height, lens,
            start=start, **options,
        )

    # One pin, taken near the rim so that "the axis moved onto it" is a long way.
    # The frame it is taken from has the attitude the drag assumed and only its
    # axis is off -- a photograph the user has dragged roughly right, which is the
    # state the first pin is placed in.
    offset = CameraModel(lens=lens, xc=0.53, yc=0.47)
    one = fit_camera_from_pins(
        pins(offset, 8)[7:], directions[7:], width, height, lens, start=start
    )
    # The star it names lands on the pixel it was given -- that is all one pair
    # can say -- and the attitude is left exactly where the drag put it.
    assert not one.measured and one.rms_px < 1e-6, (one.measured, one.rms_px)
    assert one.camera.roll == start.roll and one.camera.tilt_x == start.tilt_x, one.camera
    one_x, one_y = one.centre_px(height)
    moved = math.hypot(one_x - start.xc * height, one_y - start.yc * height)
    assert moved > 20.0, f"one pin moved the axis by only {moved:.1f} px"

    two = fit(flat, 2)
    assert two.rms_px < 1e-6, two.rms_px
    assert abs(two.camera.roll - flat.roll) < 1e-6, (two.camera.roll, flat.roll)
    assert abs(two.camera.tilt_x) < 1e-9 and abs(two.camera.tilt_y) < 1e-9, two.camera
    two_x, two_y = two.centre_px(height)
    assert math.hypot(two_x - flat.xc * height, two_y - flat.yc * height) < 1e-3, (two_x, two_y)

    three = fit(tilted, 3)
    assert three.rms_px < 1e-6, three.rms_px
    assert abs(three.camera.roll - tilted.roll) < 1e-6, (three.camera.roll, tilted.roll)
    for got, want, name in (
        (three.camera.tilt_x, tilted.tilt_x, "tilt_x"),
        (three.camera.tilt_y, tilted.tilt_y, "tilt_y"),
    ):
        assert abs(got - want) < 1e-6, (name, got, want)
    three_x, three_y = three.centre_px(height)
    assert math.hypot(three_x - tilted.xc * height, three_y - tilted.yc * height) < 1e-3, (
        three_x, three_y
    )
    # The axis is the reading that matters for an off-zenith frame: tilt is a
    # tangent-plane offset, so what has to come back is the *angle* off the zenith.
    pointing = math.degrees(float(altaz_from_enu(three.camera.rotation()[:, 2])[0]))
    wanted = 90.0 - math.degrees(math.atan(math.hypot(0.42, 0.31)))
    assert abs(pointing - wanted) < 0.02, (pointing, wanted)
    print(
        f"    one pin moves the axis {moved:.1f} px; two set the roll"
        f" ({math.degrees(two.camera.roll):.3f} deg); three set the tilt"
        f" ({math.degrees(three.camera.tilt_x):.3f} / {math.degrees(three.camera.tilt_y):.3f}"
        f" deg, pointing alt {pointing:.2f} deg)"
    )

    # A pin that disagrees is the noise the fit cannot explain, and it is the one
    # that ends up carrying it: six pins, one of them 40 px out, and the outlier's
    # residual is the largest by a factor of four.
    noisy = pins(flat, 6)
    noisy[0] = ControlPoint(noisy[0].name, noisy[0].x + 40.0, noisy[0].y)
    shaken = fit_camera_from_pins(noisy, directions[:6], width, height, lens, start=start)
    residuals = shaken.residuals
    print(
        f"    six pins, one 40 px out: residuals "
        f"{', '.join(f'{value:.1f}' for value in residuals)} px"
        f" ({shaken.rms_px:.1f} px RMS)"
    )
    assert int(np.argmax(residuals)) == 0, residuals
    assert residuals[0] > 2.0 * float(np.min(residuals)), residuals
    assert shaken.rms_px > 8.0, shaken.rms_px

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
    shown = client.get("/api/session").get_json()
    rings = shown["overlay"]["stars"]
    assert len(rings) >= 6, f"the overlay only found {len(rings)} bright stars to pin"

    # An empty list is how the page forgets its pins: the answer is the sky the
    # drag implies and nothing is left measured.
    forgotten = client.post("/api/live", json={"points": []})
    assert forgotten.status_code == 200, forgotten.get_json()
    assert forgotten.get_json()["lens"] is None and session.pins is None, forgotten.get_json()

    # A pair that is not a pair, or a star the field does not have, is a mistake on
    # the wire rather than an answer of "no change".
    assert client.post("/api/live", json={"points": [{"name": "Vega"}]}).status_code == 400
    unknown = client.post("/api/live", json={"points": [{"name": "Nonesuch", "x": 10, "y": 20}]})
    assert unknown.status_code == 400 and "Nonesuch" in unknown.get_json()["error"], (
        unknown.get_json()
    )

    # The circles the page shows are exactly where the seed camera puts those
    # stars, so pinning them back must leave the camera where it was: this is the
    # loop the user drives, with no error in it to explain.
    pins = [{"name": ring["name"], "x": ring["x"], "y": ring["y"]} for ring in rings]
    three = client.post("/api/live", json={"points": pins[:3]}).get_json()
    assert three["lens"] is None and three["rms_px"] < 0.5, three["rms_px"]
    assert abs(three["camera"]["pointing_alt_deg"] - 90.0) < 0.01, three["camera"]

    six = client.post("/api/live", json={"points": pins[:6]}).get_json()
    assert six["lens"] is not None and six["lens"]["measured"], six.get("lens_error")
    assert six["rms_px"] < 6.0, six["rms_px"]
    recovered = six["lens"]["focal_px"]
    seed_focal = session.seed_focal_px()
    assert abs(recovered - seed_focal) < 0.05 * seed_focal, (recovered, seed_focal)

    # The pairs are the pixels the user clicked, so the answer depends on the
    # pairs and not on how the photo happens to be dragged: three different
    # transforms, the same camera.  (This is what lets the page turn a frame after
    # it has been pinned without moving the calibration.)
    for alignment in (
        {"roll_deg": 37.0, "dx": 120.0, "dy": -40.0},
        {"roll_deg": -97.5, "mirrored": False, "dx": -8.0, "dy": 12.0},
    ):
        moved = client.post("/api/live", json={"points": pins[:6], **alignment}).get_json()
        for key in ("roll_deg", "tilt_x_deg", "tilt_y_deg"):
            assert abs(moved["camera"][key] - six["camera"][key]) < 1e-6, (key, moved["camera"])
        assert abs(moved["camera"]["centre_px"]["x"] - six["camera"]["centre_px"]["x"]) < 1e-6
        assert abs(moved["rms_px"] - six["rms_px"]) < 1e-6, (moved["rms_px"], six["rms_px"])

    # The line the page draws runs from the photo pixel (moved by the transform it
    # is showing) to the ring's new position, and that length is the residual.
    screen = Alignment.from_dict(six["alignment"])
    for point in six["points"]:
        foot_x, foot_y = screen.forward(point["x"], point["y"], image_w, image_h)
        drawn = math.hypot(point["draw_x"] - float(foot_x), point["draw_y"] - float(foot_y))
        assert abs(drawn - point["residual_px"]) < 1e-6, (drawn, point["residual_px"])
    print(
        f"    {len(pins[:6])} pins fitted through /api/live: {six['rms_px']:.2f} px RMS,"
        f" f {recovered:.1f} px/rad (seed {seed_focal:.1f}),"
        f" {six['lens']['num_coefficients']} k term(s),"
        f" check {session.pins.restate_px:.2f} px"
    )

    # A wrong mirror is the one thing the *fitter* can put right: the pins are a
    # mirrored star pattern, which no rotation can explain, so the answer flips the
    # flag and the page turns the photo over with it.  Two pins are not enough to
    # know (any two vectors fit a rotation either way round), which is why the
    # correction starts at MIN_PINS_FOR_MIRROR.
    flipped = client.post("/api/live", json={"points": pins[:3], "mirrored": True}).get_json()
    assert flipped["alignment"]["mirrored"] is False, flipped["alignment"]
    assert flipped["rms_px"] < 2.0, flipped["rms_px"]
    # Two pins cannot test the hand -- any two vectors fit a rotation either way
    # round -- so the toggle is simply believed, from a frame with no pins yet.
    client.post("/api/live", json={"points": []})
    kept = client.post("/api/live", json={"points": pins[:2], "mirrored": True}).get_json()
    assert kept["alignment"]["mirrored"] is True, kept["alignment"]




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
        camera=CameraModel(lens=session.seed_lens(), xc=0.5, yc=0.5),
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
        f"{gap:.1f} px from Alcyone, {moved_px:.1f} px of movement through the seed "
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
    """Four pinned stars calibrate the lens; three cannot, and seven buy k4.

    The lens is the model ``cv2.fisheye`` calibrates -- focal length plus the odd
    ``k`` terms of::

        r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)

    -- and it is solved *together* with the axis and the attitude, which is what
    lets a frame that does not point at the zenith be calibrated at all.  Four
    pins are the count at which the lens becomes solvable (the optical centre and
    ``k1``), and each further pin buys one more odd term, up to ``k4``, which is
    where ``cv2.fisheye`` stops and where the pins stop too.

    This check pins down the statements the page depends on: the counting rule,
    that the two forms of the model describe the same curve, that a synthetic
    (f, k1 .. k4) on a *tilted* camera comes back exactly -- lens, axis, roll and
    tilt together -- that a frozen focal length cannot absorb a lens that is
    simply wider than the starting curve, that fewer than four pins measure no
    lens at all, and that the artefact ``/api/save`` writes holds the camera the
    pins measured, in pixels of the photograph.
    """
    from .geometry import CameraModel, enu_from_altaz
    from .model import LensModel, fisheye_theta
    from .server import FrameStore, create_app
    from .session import (
        MIN_PINS_FOR_LENS,
        SEED_COEFFICIENTS,
        SEED_FOCAL_PX,
        ControlPoint,
        fit_camera_from_pins,
        pin_coefficient_count,
    )

    assert MIN_PINS_FOR_LENS == 4, MIN_PINS_FOR_LENS
    # Four, five, six and seven pins buy k1 .. k4; past that the order is capped,
    # so an eighth star and beyond over-determine the same unknowns.
    counts = [pin_coefficient_count(count) for count in (3, 4, 5, 6, 7, 8, 9)]
    assert counts == [1, 1, 2, 3, 4, 4, 4], counts

    # The two forms are the same family: the odd polynomial fitted to the OpenCV
    # curve reproduces it to a small fraction of a pixel.
    height, width = 1400.0, 1400.0
    focal, distortion = 520.0, (-0.012,)
    converted = LensModel.from_fisheye(focal, distortion, height_px=height, sigma_edge=0.988)
    radius = np.linspace(0.0, 0.988 * 0.5 * height, 200)
    drift = np.degrees(np.abs(converted.theta(radius / (0.5 * height))
                              - fisheye_theta(radius, focal, distortion))).max()
    assert drift < 0.02, f"the polynomial form misses the fisheye curve by {drift:.4f} deg"

    # What the pins are taken from: a frame the overlay has never seen, pointing
    # 7 deg away from the zenith so that the tilt has to be measured as well.
    directions = enu_from_altaz(
        np.radians(90.0 - np.array([84.0, 65.0, 40.0, 10.0, 84.0, 65.0, 40.0, 10.0])),
        np.radians(np.array([15.0, 195.0, 15.0, 195.0, 150.0, 30.0, 250.0, 100.0])),
    )
    ideal = LensModel.from_family("equidistant", fov_deg=180.0, sigma_edge=0.988)
    start = CameraModel(lens=ideal)

    def synthetic(f, ks):
        """Pins a tilted lens of ``(f, ks)`` would produce exactly."""
        lens = LensModel.from_fisheye(f, ks, height_px=height, sigma_edge=0.988)
        camera = CameraModel(
            lens=lens, xc=0.523, yc=0.472, roll=math.radians(23.0),
            tilt_x=0.12, tilt_y=-0.09,
        )
        x, y = camera.pixel_from_enu(directions, height)
        points = [
            ControlPoint(f"synthetic {index}", float(px), float(py))
            for index, (px, py) in enumerate(zip(x, y))
        ]
        return camera, points

    def fit(points, count, **options):
        return fit_camera_from_pins(
            points[:count], directions[:count], width, height, ideal,
            start=start, **options,
        )

    truth, pins = synthetic(focal, distortion)
    four = fit(pins, 4)
    assert four.measured and four.num_coefficients == 1, (four.measured, four.coefficients)
    assert abs(four.focal_px - focal) < 1.0, (four.focal_px, focal)
    assert abs(four.coefficients[0] - distortion[0]) < 1e-4, four.coefficients
    assert four.rms_px < 0.5, four.rms_px
    for got, want, name in (
        (four.centre_px(height)[0], truth.xc * height, "centre x"),
        (four.centre_px(height)[1], truth.yc * height, "centre y"),
        (four.camera.roll, truth.roll, "roll"),
        (four.camera.tilt_x, truth.tilt_x, "tilt x"),
        (four.camera.tilt_y, truth.tilt_y, "tilt y"),
    ):
        assert abs(got - want) < 0.05, (name, got, want)
    # Four stars only buy k1, and the readout says what to pin for the next term.
    assert any("theta^5" in note for note in four.notes), four.notes

    five = fit(pins, 5)
    assert five.num_coefficients == 2 and abs(five.coefficients[1]) < 1e-3, five.coefficients
    assert abs(five.focal_px - focal) < 1.0, five.focal_px

    # The sixth star buys the third odd term and the seventh the fourth, so a lens
    # with real theta^7 and theta^9 components comes back exactly.
    deep_focal, deep_k = 581.2, (-0.098, 0.0048, -0.00062, 0.0002)
    deep_truth, deep_pins = synthetic(deep_focal, deep_k)
    deep = fit(deep_pins, 7)
    assert deep.num_coefficients == 4, deep.num_coefficients
    assert abs(deep.focal_px - deep_focal) < 1.0, (deep.focal_px, deep_focal)
    for got, want in zip(deep.coefficients, deep_k):
        assert abs(got - want) < 1e-3, (deep.coefficients, deep_k)
    assert abs(deep.camera.tilt_x - deep_truth.tilt_x) < 1e-3, deep.camera.tilt_x
    assert deep.rms_px < 1.0, deep.rms_px
    # An eighth star is another equation for the same unknowns, and the readout
    # says which way those extra pins were spent instead of promising a k5.
    eight = fit(deep_pins, 8)
    assert eight.num_coefficients == 4, eight.num_coefficients
    capped = [note for note in eight.notes if "theta^9" in note]
    assert capped and "over-determine" in capped[0], eight.notes

    # A frozen focal length cannot absorb the starting curve's field of view being
    # off: the linear term is then nailed to it, and k1 is left describing a pure
    # scale error.  This is why the focal length is fitted.
    frozen = fit(pins, 4, optimize_focal=False)
    assert frozen.rms_px > 20.0 * max(four.rms_px, 1e-6), (frozen.rms_px, four.rms_px)
    print(
        f"    synthetic: 4 pins -> f {four.focal_px:.2f} k1 {four.coefficients[0]:.5f}"
        f" ({four.rms_px:.4f} px), 7 pins -> f {deep.focal_px:.2f}"
        f" ({deep.rms_px:.4f} px), frozen f -> {frozen.rms_px:.1f} px,"
        f" restate {four.restate_px:.2f} px"
    )

    # Below four pins there is no lens to measure -- the axis and the attitude are
    # what they determine -- and the fit says so instead of inventing a curve.
    three = fit(pins, 3)
    assert not three.measured and three.focal_px is None, (three.measured, three.focal_px)
    assert three.lens is start.lens, three.lens
    assert any("place the axis" in note for note in three.notes), three.notes
    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the lens round trip")
        return
    store = FrameStore(calibration_dir=Path(tempfile.mkdtemp()))
    client = create_app(store).test_client()
    assert client.post("/api/open", json={"path": str(sample)}).status_code == 200
    session = store.session
    width_px, height_px = float(session.width), float(session.height)
    shown = client.get("/api/session?projection=fisheye&fov=180&mag=4.5").get_json()
    rings = shown["overlay"]["stars"]

    # The real frame with a made-up photograph: the stars are a *known* camera's,
    # 7 deg off the zenith, so the fit has an answer it can be held to.  That is
    # the whole wire path -- /api/live, the camera fit, the re-projected overlay
    # and /api/save -- without needing a solved camera to invent the truth.
    index = {str(name): position for position, name in enumerate(session.field.names)}
    chosen = [ring for ring in rings if ring["name"] in index][:7]
    assert len(chosen) == 7, f"only {len(chosen)} of the rings are in the field"
    directions = np.array([session.field.enu[index[ring["name"]]] for ring in chosen])
    truth_focal, truth_k = 560.0, (-0.098, 0.0048, -0.00062)
    truth = CameraModel(
        lens=LensModel.from_fisheye(
            truth_focal, truth_k, height_px=height_px, sigma_edge=session.circle.sigma_edge
        ),
        xc=0.52, yc=0.47, roll=math.radians(12.0), tilt_x=0.10, tilt_y=-0.07,
    )
    click_x, click_y = truth.pixel_from_enu(directions, height_px)
    body = {
        "projection": "fisheye",
        "fov_deg": 180.0,
        "mag_limit": 4.5,
        "points": [
            {"name": ring["name"], "x": float(px), "y": float(py)}
            for ring, px, py in zip(chosen, click_x, click_y)
        ],
    }
    reply = client.post("/api/live", json=body)
    assert reply.status_code == 200, reply.get_data(as_text=True)[:300]
    fitted = reply.get_json()
    camera = fitted["camera"]
    assert fitted["lens"] is not None and fitted["lens"]["measured"], fitted.get("lens_error")
    assert fitted["lens"]["num_coefficients"] == 4, fitted["lens"]["num_coefficients"]
    assert abs(camera["roll_deg"] - 12.0) < 0.5, camera["roll_deg"]
    assert abs(camera["tilt_x_deg"] - math.degrees(0.10)) < 0.5, camera["tilt_x_deg"]
    assert abs(camera["centre_px"]["x"] - 0.52 * width_px) < 1.0, camera["centre_px"]
    assert abs(camera["centre_px"]["y"] - 0.47 * height_px) < 1.0, camera["centre_px"]
    wanted_alt = 90.0 - math.degrees(math.atan(math.hypot(0.10, 0.07)))
    assert abs(camera["pointing_alt_deg"] - wanted_alt) < 0.5, camera["pointing_alt_deg"]
    assert abs(fitted["lens"]["focal_px"] - truth_focal) < 10.0, fitted["lens"]["focal_px"]
    assert fitted["rms_px"] < 2.0, fitted["rms_px"]
    print(
        f"    sample frame: 7 pins -> f {fitted['lens']['focal_px']:.1f} px/rad"
        f" (truth {truth_focal:.1f}), roll {camera['roll_deg']:.2f} deg,"
        f" pointing alt {camera['pointing_alt_deg']:.2f} deg,"
        f" {fitted['rms_px']:.2f} px rms"
    )

    # The artefact is the camera the pins measured, in pixels of the photograph: no
    # composition with the drag, because the fit never read it.
    saved = client.post("/api/save", json=body)
    assert saved.status_code == 200, saved.get_json()
    artefact = json.loads(Path(saved.get_json()["path"]).read_text())
    assert artefact["format"] == "lens_calibration/2", artefact["format"]
    assert artefact["lens_source"] == "pins", artefact["lens_source"]
    assert artefact["pins"]["measured"] is True, artefact["pins"]
    for key in ("roll_deg", "tilt_x_deg", "tilt_y_deg", "pointing_alt_deg"):
        assert abs(artefact["camera"][key] - camera[key]) < 1e-6, (key, artefact["camera"][key])
    assert abs(artefact["camera"]["centre_px"]["x"] - camera["centre_px"]["x"]) < 1e-6
    assert abs(
        LensModel.from_dict(artefact["lens"]).theta_edge_deg
        - session.pins.lens.theta_edge_deg
    ) < 1e-3, artefact["lens"]
    # Artefacts written before the series carried sigma^9 must still load: the
    # term they never had is zero, not an error.
    legacy = LensModel.from_dict(dict(artefact["lens"], coeffs=artefact["lens"]["coeffs"][:4]))
    assert len(legacy.coeffs_deg) == 5 and legacy.coeffs_deg[4] == 0.0, legacy.coeffs_deg
    print(f"    saved {Path(saved.get_json()['path']).name}: {saved.get_json()['bytes']} bytes")

    # Undoing a pin drops back below the threshold, so the lens goes with it -- and
    # the artefact refuses to be written at all.
    back = client.post("/api/live", json={**body, "points": body["points"][:3]}).get_json()
    assert back["lens"] is None and not back["lens_error"], back.get("lens_error")
    assert session.pins is not None and not session.pins.measured
    refused = client.post("/api/save", json={**body, "points": body["points"][:3]})
    assert refused.status_code == 400, refused.get_json()
    assert "no lens measured" in refused.get_json()["error"], refused.get_json()




def check_calibration_beside_frame() -> None:
    """The artefact is written beside the photograph it describes.

    A folder of calibrations keyed by file name is a second place to keep in step
    with the frames: rename a frame and its JSON no longer names it, copy a
    sequence and the JSONs stay behind.  The save now goes next to the file --
    same folder, same stem -- unless a folder is asked for explicitly.  An upload
    is the exception the browser forces: the server only ever has its own copy
    under the temporary directory, so writing there would lose the file, and it
    goes where saves have always gone instead.
    """
    from .server import UPLOAD_CALIBRATION_DIR, UPLOAD_DIR, FrameStore, create_app

    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the save-path check")
        return
    folder = Path(tempfile.mkdtemp())
    try:
        photo = cv2.resize(
            cv2.imread(str(sample), cv2.IMREAD_COLOR), (600, 600),
            interpolation=cv2.INTER_AREA,
        )
        frame = folder / "night.jpg"
        assert cv2.imwrite(str(frame), photo), frame

        store = FrameStore(work_height=200)
        session = store.load(frame)
        assert session.calibration_dir is None, session.calibration_dir
        assert session.calibration_path() == folder / "night.json", session.calibration_path()
        written = session.save_calibration()
        assert written.is_file() and written.parent == folder, written
        payload = json.loads(written.read_text())
        assert payload["image"]["path"] == str(frame), payload["image"]["path"]
        assert session.summary()["calibration_path"] == str(written)

        # An explicit folder still wins -- that is what --calibration-dir is for.
        elsewhere = folder / "artefacts"
        override = FrameStore(work_height=200, calibration_dir=elsewhere)
        assert override.load(frame).calibration_path() == elsewhere / "night.json"

        # A browser upload has no folder of its own to write back to.
        uploaded = FrameStore(work_height=200)
        client = create_app(uploaded).test_client()
        posted = client.post(
            "/api/open",
            data={"file": (io.BytesIO(frame.read_bytes()), "uploaded.jpg")},
            content_type="multipart/form-data",
        )
        assert posted.status_code == 200, posted.get_json()
        assert Path(uploaded.session.path).parent == UPLOAD_DIR, uploaded.session.path
        assert uploaded.session.calibration_path().parent == UPLOAD_CALIBRATION_DIR, (
            uploaded.session.calibration_path()
        )
        assert client.get("/api/session").get_json()["calibration_path"].endswith(
            "uploaded.json"
        )
        print(
            f"    {frame.name} -> {written.name} beside it;"
            f" an upload -> {UPLOAD_CALIBRATION_DIR}/uploaded.json"
        )
    finally:
        (UPLOAD_DIR / "uploaded.jpg").unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)


def check_corrected_preview() -> None:
    """The corrected picture is the ideal equiangular lens, at the frame's size.

    The preview exists to be *flashed* against the photograph, so four things
    have to be true or it would lie in a plausible-looking way:

    1. it is a resample of the measured camera onto the ideal one.  A synthetic
       star at a known field angle lands where the equiangular curve puts that
       angle, on the same azimuth -- the correction is a radial stretch about the
       optical axis and nothing else;
    2. a camera that is already ideal is left alone: the identity is the strongest
       statement that the map is the lens and not a second opinion;
    3. the raster is the photograph's own size, and the frame is still the frame:
       a resample that changed the size would make the page's overlay wrong;
    4. the overlay asked for with ``corrected=1`` is drawn through that same ideal
       lens.  On the photograph the rings must mark where the *photograph* puts
       the stars; on the corrected picture they must mark where the ideal lens
       does.  Drawing them through the measured lens would show the correction
       itself as an error -- the whole feature, inverted.
    """
    from .geometry import remap_to_ideal
    from .server import FrameStore, create_app
    from .session import CORRECTED_URL

    sample = Path(__file__).resolve().parent.parent / "sampleimages" / "00000687.jpg"
    if not sample.is_file():
        print(f"    {sample.name} not found: skipping the corrected preview check")
        return
    folder = Path(tempfile.mkdtemp())
    try:
        height = width = 600.0
        photo = cv2.resize(
            cv2.imread(str(sample), cv2.IMREAD_COLOR), (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        )
        frame = folder / "night.jpg"
        assert cv2.imwrite(str(frame), photo), frame

        # --- 1. the star goes where the ideal lens puts it ------------------
        # The stand-in lens is an equisolid disc, whose rim is compressed by a
        # good fraction of a pixel per pixel against the equiangular ideal: a
        # correction of it is easy to see, and a map that did nothing would fail.
        camera = CameraModel(lens=stand_in_lens(), xc=0.505, yc=0.497)
        ideal = camera.ideal_equiangular()
        assert ideal.lens.sigma_edge == camera.lens.sigma_edge, ideal.lens.sigma_edge
        assert abs(ideal.lens.theta_edge_deg - camera.lens.theta_edge_deg) < 1e-9, (
            "the ideal lens moved the rim angle"
        )
        # Equiangular is one odd power: theta proportional to sigma, exactly.
        sigma = np.array([0.2, 0.5, 0.9])
        rate = ideal.lens.theta(sigma) / sigma
        assert np.allclose(rate, rate[0], rtol=0.0, atol=1e-12), rate

        canvas = np.zeros((int(height), int(width), 3), np.uint8)
        theta_want = math.radians(55.0)
        radius_src = float(camera.lens.sigma(np.array([theta_want]))[0]) * height / 2.0
        centre_x, centre_y = camera.centre_px(height)
        azimuth = math.radians(35.0)
        x0 = int(round(centre_x + radius_src * math.cos(azimuth)))
        y0 = int(round(centre_y - radius_src * math.sin(azimuth)))
        canvas[y0 - 2:y0 + 3, x0 - 2:x0 + 3] = 255

        corrected = remap_to_ideal(canvas, camera, ideal)
        assert corrected.shape == canvas.shape, corrected.shape
        assert corrected[0, 0].max() == 0, "a corner outside the disc is not black"

        def star_at(image: np.ndarray) -> tuple[float, float]:
            ys, xs = np.nonzero(image.max(axis=2) > 40)
            assert xs.size, "the synthetic star did not survive the resample"
            return float(xs.mean()), float(ys.mean())

        got_x, got_y = star_at(corrected)
        radius_out = math.hypot(got_x - centre_x, centre_y - got_y)
        want = float(ideal.lens.sigma(np.array([theta_want]))[0]) * height / 2.0
        assert abs(radius_out - want) <= 2.0, (
            f"the corrected star sits at {radius_out:.1f} px, the ideal lens says {want:.1f}"
        )
        got_azimuth = math.degrees(math.atan2(centre_y - got_y, got_x - centre_x))
        assert abs(got_azimuth - 35.0) < 1.5, (
            f"the correction moved the azimuth to {got_azimuth:.1f} deg"
        )

        # --- 2. correcting an ideal camera is the identity ------------------
        plain_x, plain_y = star_at(remap_to_ideal(canvas, ideal, ideal))
        assert abs(plain_x - x0) <= 1.5 and abs(plain_y - y0) <= 1.5, (
            f"an already ideal camera moved the star from ({x0}, {y0}) to"
            f" ({plain_x:.1f}, {plain_y:.1f})"
        )

        # --- 3. the endpoint renders it at the photograph's own size --------
        store = FrameStore(work_height=200)
        store.load(frame)
        store.session.configure(
            utc=dt.datetime(2026, 1, 18, 9, 15, 17, tzinfo=dt.timezone.utc),
            latitude=64.5724,
            longitude=-149.262,
        )
        client = create_app(store).test_client()
        query = "projection=equisolid&fov=180"
        rendered = client.get(f"{CORRECTED_URL}?{query}")
        assert rendered.status_code == 200, rendered.get_json()
        assert rendered.mimetype == "image/png", rendered.mimetype
        decoded = cv2.imdecode(np.frombuffer(rendered.data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == photo.shape, (
            f"{None if decoded is None else decoded.shape} is not the photograph's {photo.shape}"
        )
        # An equisolid disc is not equiangular, so the pixels have to have moved:
        # a render identical to the file would be a preview of nothing.
        moved = float(np.abs(decoded.astype(np.float64) - photo.astype(np.float64)).mean())
        assert moved > 0.5, f"the corrected render differs from the frame by only {moved:.2f}"
        other = client.get(f"{CORRECTED_URL}?projection=equisolid&fov=190")
        assert other.status_code == 200 and other.data != rendered.data, (
            "a different lens produced the same corrected picture"
        )
        assert create_app(FrameStore()).test_client().get(CORRECTED_URL).status_code == 409, (
            "a server with no frame did not refuse to render one"
        )

        # --- 4. the overlay is drawn through the same ideal lens ------------
        plain = client.get(f"/api/session?{query}").get_json()
        switched = client.get(f"/api/session?{query}&corrected=1").get_json()
        assert plain["corrected"]["url"] == CORRECTED_URL, plain["corrected"]
        assert plain["corrected"]["projection"] == "equiangular", plain["corrected"]
        assert plain["corrected"]["active"] is False, plain["corrected"]
        assert switched["corrected"]["active"] is True, switched["corrected"]
        before, after = plain["overlay"], switched["overlay"]
        assert before and after
        assert abs(before["radius_px"] - after["radius_px"]) < 1e-6, (
            "the corrected overlay moved the disc rim"
        )
        where = {star["name"]: (star["x"], star["y"]) for star in after["stars"]}
        shifts = [
            math.hypot(star["x"] - where[star["name"]][0], star["y"] - where[star["name"]][1])
            for star in before["stars"] if star["name"] in where
        ]
        assert shifts and max(shifts) > 1.0, (
            "the corrected overlay is the measured one, so the correction would"
            " look like an error"
        )
        print(
            f"    ideal lens: a 55 deg star moves {radius_src - radius_out:+.1f} px;"
            f" rings shift up to {max(shifts):.1f} px; {len(rendered.data)} byte PNG"
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#: Run in order; each one is a statement the rest of the package relies on.
CHECKS = (
    check_rotation,
    check_sky_orientation,
    check_pixel_handedness,
    check_round_trip,
    check_web_ui,
    check_frame_picker,
    check_live_alignment,
    check_pinned_lens,
    check_sky_aids,
    check_field_for_today,
    check_calibration_beside_frame,
    check_corrected_preview,
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
