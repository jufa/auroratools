"""Per-image state for the interactive web UI.

One :class:`AlignmentSession` owns everything that is expensive or stateful
about a single exposure -- the EXIF metadata, the downscaled work raster, the
image-circle fit and the catalogue field -- so a web request never has to reload
the frame.

The overlay is a fisheye *of the rig*, not an abstract reference: the bright
catalogue stars are ringed where a camera with :data:`SEED_FOCAL_PX` pixels per
radian, :data:`SEED_COEFFICIENTS` and the seed's optical centre puts them, and
the first two pins replace that camera with the one the frame actually had --
lens, optical centre, roll, the tilt of its axis and its mirror, all measured.
Starting from the measured lens rather than from an ideal equiangular disc is
what makes the first drag land close: the rings are already where the stars are,
so the residual the user is chasing is the lens error and not the difference
between two projections.

Before the first pins the drag *is* the statement about where the sky is
(:meth:`AlignmentSession.camera_for`); afterwards it is only a way of looking at
the frame, because the camera has been measured in the frame's own pixels and
both the rings and the calibration are drawn from it.

There is no automatic solver.  The stars are found by the person looking at the
frame, which is both more accurate than a ballot of detections and the only
thing that works on a frame whose stars are few, trailed, or half hidden by
aurora -- and the answer needs no search, because the pins say which star is
which.

What the page is shown is the photograph.  ``/frame.jpg`` hands over the file's
own pixels at the size they were taken at -- byte for byte when it is a JPEG
that carries no EXIF orientation -- and never a stretched or downscaled copy: a
night frame is nearly black, but lifting it is a *view*, so the page does it
with one gamma curve that starts flat (see :func:`_frame_bytes`).  The work
raster is the geometry's reference, not a display limit; leaning in with the
browser's own zoom reaches the stars the sensor recorded.

A second picture the page can ask for is ``/corrected.png``
(:meth:`AlignmentSession.corrected_png`): the same frame resampled onto an ideal
equidistant lens through the camera the pins measure, at full resolution and the
same size, so swapping it under the overlay is a flash comparison between the
lens' actual distortion and the ideal one.  It is a *view* of the same
measurement -- rendered on demand, cached until the camera moves, and never
written anywhere.

Why the conversion is done numerically
-------------------------------------
A mirror/rotation/translation of the raster is a 2-D orthogonal map composed
with the overlay's radial map, so it is exactly representable by another
:class:`CameraModel` -- but only if the sign conventions of the web transform,
the pixel convention and the roll are all right.  Rather than derive those
signs (the historical source of the "everything solves mirrored" bug), the
transform is applied to a ring of known sky directions and the attitude is
recovered with the same closed-form Kabsch update the calibration itself uses.
Mirroring is cross-checked by the improper-rotation test inside
:func:`~lens_calibration.match.update_attitude`.

Pinning stars instead of dragging
---------------------------------
Dragging lines the whole sky up by eye; it cannot say *which* star the user is
looking at.  The page therefore also lets the user pin a star: click its circle
in the overlay, then click the same star in the photograph, and
:func:`fit_camera_from_pins` refits the camera through every pair at once.  That
is ordinary georeferencing, and it is what turns the manual step into a
control-point problem with an exact answer instead of an eyeballed one.

Every pair is (a named star, a pixel of the photograph).  The star is named, so
its field angle and its place in the sky come from the catalogue, and no part of
the fit mentions the drag: the transform only decides *where on screen* the
frame and the sky are drawn, never what the camera is.  Dragging a photo after
it has been pinned therefore cannot move the calibration, only the view.

What the pairs determine grows one degree of freedom at a time, and every stage
is the closed-form answer to a smaller question:

* one pin places the optical axis (two numbers from two equations);
* two pins turn the sky as well: the rotation carrying the clicked pixels onto
  their stars is Kabsch's, in closed form
  (:func:`~lens_calibration.match.rotation_from_pairs`);
* three pins let the axis and the attitude settle together, and three is also
  where a pin that disagrees becomes visible instead of being absorbed;
* from :data:`MIN_PINS_FOR_LENS` pins the radial model joins in: the optical
  centre, the focal length and the odd ``k`` coefficients of the fisheye curve
  ``cv2.fisheye`` calibrates -- four pins buy ``k1``, and each further pin one
  more odd term (the fifth ``k2``, the sixth ``k3``, the seventh ``k4``), and
  there the polynomial order stops (:data:`MAX_PIN_ORDER_DEG`), so every further
  star is another equation for the same unknowns rather than another coefficient.

Attitude and lens are coupled -- the field angle of a pixel is read through the
curve and the curve is fitted to those angles -- which is why the two are solved
*together*: one :func:`scipy.optimize.least_squares` over the attitude, the
optical centre and the radial terms at once, started from Kabsch's closed form.
Alternating them instead (Kabsch, then the radial half, repeat) looks cheaper
and is not: a radius error is read as a tilt, so each round hands the other a
compensating error and the pair can stall well short of the truth.

The tilt is the reason a frame that does not point at the zenith now works.  A
tilted camera sees the sky squashed on one side, which no shift and no rotation
of the raster can imitate: the old alignment had three degrees of freedom and a
tilted frame needs five.  The pins now measure all five, plus the mirror: a
raster that is a mirror image of the sky shows up as an improper Kabsch
rotation, and from :data:`MIN_PINS_FOR_MIRROR` pins the fit corrects it instead
of trusting the toggle.

Drawing help that is never fitted
---------------------------------
Clusters, nebulae and the Milky Way are drawn beside the rings
(:meth:`AlignmentSession._aid_layer`, fed by :mod:`lens_calibration.aids`), in
their own colour and from their own table.  They are orientation cues -- which
smudge is the Pleiades -- and deliberately not control points: a pin has to hang
off a star, whose position is a point.  Keeping them out of ``overlay["stars"]``
is what makes that true on the page too, because the hit test only walks the
rings.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .aids import sky_aids
from .catalog import StarField, star_field
from .exifmeta import ExposureMeta, read_exposure_meta
from .geometry import (
    CameraModel,
    altaz_from_enu,
    attitude_from_rotation,
    enu_from_altaz,
    remap_to_ideal,
)
from .match import (
    CircleFit,
    Correspondence,
    DetectedStars,
    fit_image_circle,
    rotation_from_pairs,
    update_attitude,
)
from .model import (
    FAMILIES,
    LensModel,
    equiangular_pixels_per_radian,
    fisheye_fold_radius,
    fisheye_radius,
)

LOGGER = logging.getLogger(__name__)

#: Height of the raster the *geometry* is done at.  The lens model is
#: dimensionless, so a calibration fitted here transfers to any raster size; this
#: is only the size the seed focal length in :data:`SEED_FOCAL_PX` is quoted at.
#: It is not a display limit: the page is served the photograph itself, at the
#: resolution it was taken at, and only the pins and the overlay are in raster
#: pixels.
WORK_HEIGHT = 1400

#: Catalogue limit for the overlay rings: bright enough to recognise patterns,
#: few enough to keep the overlay readable.
OVERLAY_MAG_LIMIT = 4.5

#: Where the page fetches the corrected preview (the photograph resampled onto an
#: ideal equiangular lens).  It is rendered on demand, at the photograph's own
#: size, so the page can swap it under the same overlay for a flash comparison;
#: the route and the URL in the payload are one spelling (:data:`CORRECTED_URL`).
CORRECTED_URL = "/corrected.png"

#: Projection the overlay is drawn with unless the user says otherwise: the
#: lens this rig measures, so the circles start out on the stars.
SEED_PROJECTION = "fisheye"

#: The lens this rig's pins measure: the focal length in pixels per radian of the
#: work raster, the first ``k`` terms of
#:
#:     r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6)      t = theta
#:
#: and the optical centre in pixels of the same raster.  The centre is worth
#: carrying: it sits a few pixels below the middle of the frame, and a pin's
#: radius -- so its field angle -- is read from it, so a seed centre that
#: disagrees with the rig tilts every radius the fit starts from.
#:
#: ``k3`` starts at zero because nothing has measured it yet: two terms were as
#: far as the pins went, and a seventh star is what buys the fourth.  These are
#: *starting conditions*, not a calibration: they put the overlay within a couple
#: of percent of the real star pattern, which is what makes the first drag and
#: the first pins land close.  The pins replace them (and the page lets them be
#: typed in), so being off here costs iterations, not truth.
SEED_FOCAL_PX = 582.3
SEED_COEFFICIENTS: tuple[float, ...] = (-0.10455, -0.00080, 0.0)
SEED_CENTRE_PX: tuple[float, float] = (700.1, 703.7)

#: Most ``k`` terms an overlay or a pinned fit may carry: ``cv2.fisheye``'s
#: ``k1 .. k4``.  A seventh pinned star is what buys ``k4``; past it more stars
#: add equations rather than coefficients.
MAX_COEFFICIENTS = 4


#: Everything the overlay can be drawn with: the measured lens (``fisheye``) or
#: one of the analytic projections.
PROJECTIONS: tuple[str, ...] = (SEED_PROJECTION,) + FAMILIES

#: Zenith distances of the alt-az grid rings drawn under the star circles.
GRID_THETA_DEG = (30.0, 60.0, 90.0)

#: Samples per grid ring and per meridian.  The rings are projected point by
#: point through the camera (a tilted one does not draw a zenith-distance ring as
#: a circle), so this is what decides how smooth a curve looks: three degrees of
#: great circle per sample is a chord sagging by a quarter of a pixel at the rim
#: of a 1400 px frame, which is under the width of the line itself.
GRID_SAMPLES = 120

#: Pinned stars needed before the optical centre is solved.  One pin is two
#: equations for two unknowns, which is exactly determined -- the axis moves onto
#: the star -- and it is the stage that answers a click immediately.
MIN_PINS_FOR_LENS = 4

#: Pinned stars needed before the *attitude* is solved as well.  Two pins are
#: four equations for the three angles of a rotation: Kabsch's closed form, and
#: one equation left over to say whether the pins agree.  A single pin cannot
#: turn anything, so it leaves the attitude where the fit started (the drag).
MIN_PINS_FOR_ATTITUDE = 2

#: Pinned stars needed before the *mirror* is corrected by the fit rather than
#: taken from the page's toggle.  Two pairs fit any similarity exactly, and two
#: vectors give a Kabsch rotation whatever hand they are in, so the improper test
#: needs a third pair before it means anything.
MIN_PINS_FOR_MIRROR = 3

#: Camera pointing beyond which the (roll, tilt) attitude stops being a usable
#: description: ``tilt = tan(angle from the zenith)``, so a frame whose optical
#: axis is this far off the zenith cannot be written down.  A zenith fisheye is
#: never pointed anywhere near it; the bound is here to turn "the pins are
#: nonsense" into a sentence instead of an infinite number.
PIN_TILT_LIMIT_DEG = 85.0

#: How far the optimiser may turn the attitude away from the closed-form Kabsch
#: answer, in radians.  Kabsch is the best rotation for the lens it was given, so
#: the correction the joint fit has to make is a small one; the bound is what
#: stops a pin on the wrong star from being explained by pointing the camera at
#: the ground instead of by reporting the pin.
PIN_ATTITUDE_STEP = 0.5

#: Highest odd power the hand-pinned fit may use::
#:
#:     r = f * (theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9)
#:
#: ``cv2.fisheye`` itself stops at the same term.  The pins reach it one term at
#: a time, so a seventh star is what buys ``k4``; past this order more pins are
#: *redundancy* -- they add equations, and so tighten the fit, rather than
#: coefficients.  The loop in :func:`fit_camera_from_pins` still drops a term the
#: data cannot support, because a curve fitted past its evidence folds over just
#: short of the rim.
MAX_PIN_ORDER_DEG = 9

#: Distortion coefficients that order leaves free: ``k1`` .. ``k4``.
MAX_PIN_COEFFICIENTS = (MAX_PIN_ORDER_DEG - 1) // 2

#: How far the fitted optical centre may sit from the middle of the frame, as a
#: fraction of the frame size.  Every rig puts the disc centre within a few
#: percent of the middle; without the bound a four-pin fit trades centre against
#: cubic distortion and can produce a lens that folds over at the rim.
PIN_CENTRE_LIMIT = 0.15

#: Pins confined to the inner field leave the rim as extrapolation.  The notes
#: say so when the outermost pin is inside this radius.
PIN_COVERAGE_WARN_SIGMA = 0.6



@dataclass(frozen=True)
class Alignment:
    """How the photograph is placed on screen, for the eye and for the pointer.

    The numbers are exactly the CSS transform the UI applies to the frame
    (``translate(dx, dy) rotate(roll) scaleX(+-1)`` about the frame centre):
    ``roll_deg`` turns the raster clockwise on screen, ``mirrored`` flips it
    left to right, and ``dx``/``dy`` shift it -- all in pixels of the work
    raster.

    This is a *view*: it says where the photo is drawn, not where the sky is.
    The camera (lens, optical centre, attitude) is measured by the pins and is
    independent of it, so the overlay is drawn through the camera and then
    through this transform -- which is what keeps the rings on the stars when the
    photo is turned after it has been pinned.

    Before the first two pins there is no attitude to draw with, so the session
    reads this transform as the user's statement about where the sky is
    (:meth:`AlignmentSession.camera_for`); that camera is the starting point of
    the pin fit, and the fit replaces it.
    """

    roll_deg: float = 0.0
    mirrored: bool = False
    dx: float = 0.0
    dy: float = 0.0

    def with_mirror(self, mirrored: bool) -> "Alignment":
        """Copy with the mirror flag replaced (the fit may correct it)."""
        return Alignment(
            roll_deg=self.roll_deg, mirrored=bool(mirrored), dx=self.dx, dy=self.dy
        )

    def to_dict(self) -> dict:
        """JSON-ready summary for the UI."""
        return {
            "roll_deg": float(self.roll_deg),
            "mirrored": bool(self.mirrored),
            "dx": float(self.dx),
            "dy": float(self.dy),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "Alignment":
        """Rebuild an alignment from a request payload (missing keys = identity)."""
        payload = payload or {}
        return cls(
            roll_deg=float(payload.get("roll_deg", 0.0)),
            mirrored=bool(payload.get("mirrored", False)),
            dx=float(payload.get("dx", 0.0)),
            dy=float(payload.get("dy", 0.0)),
        )

    def _rotation(self) -> tuple[float, float]:
        angle = math.radians(self.roll_deg)
        return math.cos(angle), math.sin(angle)

    def forward(self, x, y, width: float, height: float):
        """Raster pixel position(s) -> screen overlay position(s)."""
        cos_a, sin_a = self._rotation()
        cx, cy = 0.5 * width, 0.5 * height
        u = np.asarray(x, dtype=np.float64) - cx
        v = np.asarray(y, dtype=np.float64) - cy
        if self.mirrored:
            u = -u
        return (
            u * cos_a - v * sin_a + cx + self.dx,
            u * sin_a + v * cos_a + cy + self.dy,
        )

    def inverse(self, x, y, width: float, height: float):
        """Screen overlay position(s) -> raster pixel position(s)."""
        cos_a, sin_a = self._rotation()
        cx, cy = 0.5 * width, 0.5 * height
        u = np.asarray(x, dtype=np.float64) - self.dx - cx
        v = np.asarray(y, dtype=np.float64) - self.dy - cy
        u, v = u * cos_a + v * sin_a, -u * sin_a + v * cos_a
        if self.mirrored:
            u = -u
        return u + cx, v + cy


@dataclass(frozen=True)
class ControlPoint:
    """One star the user pinned by hand.

    ``name`` is the catalogue star -- that is what the fit looks the sky
    direction up by, so a pin says *which* star and not merely where a speck is
    -- and ``x``/``y`` are pixels of the photograph: the place the user found
    it.

    Nothing here is in screen pixels.  The page undoes its own transform before
    a pin is posted, which is what makes the feed independent of how the photo
    happens to be dragged and what stops a refit from turning the photo again on
    every click.
    """

    name: str
    x: float
    y: float

    def to_dict(self, residual_px: float = 0.0, draw: tuple[float, float] | None = None) -> dict:
        """JSON-ready pair; ``residual_px`` is what the refit still misses.

        ``draw`` is where the ring is drawn on screen *now* -- a display
        position, not a measurement: it moves with the drag while ``x``/``y`` and
        the residual do not.
        """
        payload = {
            "name": self.name,
            "x": float(self.x),
            "y": float(self.y),
            "residual_px": float(residual_px),
        }
        if draw is not None:
            payload["draw_x"] = float(draw[0])
            payload["draw_y"] = float(draw[1])
        return payload


@dataclass(frozen=True)
class PinnedCamera:
    """The camera solved from the stars the user pinned by hand.

    ``camera`` is the whole answer, and the only part downstream needs: lens,
    optical centre, attitude (the ``roll`` of the frame and the ``tilt`` of its
    axis away from the zenith) and the mirror -- all of it in pixels of the work
    raster.  Everything else here is either the same answer restated in the form
    ``cv2.fisheye`` speaks (``focal_px`` and the odd ``k`` coefficients) or the
    evidence it was solved from (``points``, ``predicted``, ``residuals``).

    ``measured`` says whether the *lens* came from the pins.  Below
    :data:`MIN_PINS_FOR_LENS` pins the axis, the attitude and the mirror are
    measured but the radial curve is the one the fit started from, and a file
    that said "pins" there would be claiming a lens nobody measured.

    ``restate_px`` is the price of two languages: the fit solves ``(f, k...)``
    while the overlay draws with the package's ``theta(sigma)`` series, and at
    the rim of a strongly compressed fisheye the two bases differ by a couple of
    pixels.
    """

    camera: CameraModel
    measured: bool
    #: The pins fit an *improper* rotation: the raster they were clicked on is a
    #: mirror image of the sky as it stands.  Reported, not acted on -- see
    #: :meth:`AlignmentSession.live_alignment`.
    improper: bool
    focal_px: float | None
    coefficients: tuple[float, ...]
    rms_px: float
    max_px: float
    rms_deg: float
    covers_sigma: float
    restate_px: float
    notes: tuple[str, ...]
    points: tuple[ControlPoint, ...]
    predicted: tuple[tuple[float, float], ...]
    residuals: tuple[float, ...]

    @property
    def lens(self) -> LensModel:
        """The radial curve the overlay draws with."""
        return self.camera.lens

    @property
    def pins(self) -> int:
        """How many stars were pinned."""
        return len(self.points)

    @property
    def num_coefficients(self) -> int:
        """Number of ``k`` terms the pins determined."""
        return len(self.coefficients)

    def centre_px(self, height_px: float) -> tuple[float, float]:
        """Optical centre in pixels of a raster ``height_px`` high."""
        x, y = self.camera.centre_px(float(height_px))
        return float(x), float(y)

    def to_dict(self, height_px: float) -> dict:
        """JSON-ready representation (the UI readout and the calibration artefact)."""
        centre_x, centre_y = self.centre_px(height_px)
        alt, az = altaz_from_enu(self.camera.rotation()[:, 2])
        return {
            "pins": self.pins,
            "measured": bool(self.measured),
            "focal_px": None if self.focal_px is None else float(self.focal_px),
            "coefficients": [float(k) for k in self.coefficients],
            "coefficients_unit": (
                "k1 .. kn in r = f*theta*(1 + k1 t^2 + k2 t^4 + ...), theta in radians"
            ),
            "num_coefficients": self.num_coefficients,
            "centre_px": {"x": centre_x, "y": centre_y},
            "roll_deg": float(math.degrees(self.camera.roll)),
            "tilt_x_deg": float(math.degrees(self.camera.tilt_x)),
            "tilt_y_deg": float(math.degrees(self.camera.tilt_y)),
            "pointing_alt_deg": float(math.degrees(alt)),
            "pointing_az_deg": float(math.degrees(az)),
            "mirrored": bool(self.camera.mirrored),
            "rms_px": float(self.rms_px),
            "max_px": float(self.max_px),
            "rms_deg": float(self.rms_deg),
            "covers_sigma": float(self.covers_sigma),
            "restate_px": float(self.restate_px),
            "notes": list(self.notes),
            "points": [
                point.to_dict(residual, (drawn[0], drawn[1]))
                for point, residual, drawn in zip(self.points, self.residuals, self.predicted)
            ],
            "lens": self.camera.lens.to_dict(),
        }


def pin_coefficient_count(
    pin_count: int, max_coefficients: int = MAX_PIN_COEFFICIENTS
) -> int:
    """Distortion terms a given number of pinned stars determines.

    Four pins buy the optical centre and ``k1`` (the theta^3 term); the fifth
    buys ``k2`` (theta^5), the sixth ``k3`` (theta^7) and the seventh ``k4``
    (theta^9), which is as far as the fit goes.  The rule is the one the count of
    unknowns gives: two centre coordinates plus ``n`` coefficients against two
    equations per pin, with the pins taken in the order they were placed -- so
    one star past the count that makes the fit determined is what makes it
    *redundant* as well.  From there on the count is capped at
    :data:`MAX_PIN_COEFFICIENTS`, and every further pin is another equation for
    the same unknowns.
    """
    return max(1, min(int(pin_count) - (MIN_PINS_FOR_LENS - 1), int(max_coefficients)))


def fit_camera_from_pins(
    points: Sequence[ControlPoint],
    directions: np.ndarray,
    width: float,
    height: float,
    ideal_lens: LensModel,
    *,
    start: CameraModel | None = None,
    coefficients: int | None = None,
    optimize_focal: bool = True,
) -> PinnedCamera:
    """Solve the camera from the pinned stars: axis, attitude, lens, mirror.

    Each pin is a *named* star and a pixel of the photograph, so the fit needs no
    assumption about where the photo sits on screen: the sky direction comes from
    the catalogue and the pixel is the evidence.  What the answer contains grows
    with the number of pins, and each stage is the closed-form answer to a
    smaller question -- the axis from one pin, the attitude from two, the two
    together from three, and from :data:`MIN_PINS_FOR_LENS` pins the radial model
    as well::

        r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)

    which is what ``cv2.fisheye`` calibrates.  The coefficients enter linearly in
    the radius, so a further pin buys another odd term up to
    :data:`MAX_PIN_ORDER_DEG`; past that more pins over-determine the same
    unknowns, which is the point -- the fit averages their noise instead of
    chasing it.

    Attitude and lens are solved in turn rather than in one go because they are
    coupled but not equally well conditioned: the attitude is a rotation, which
    Kabsch settles exactly in one step, while the radial half is a bounded least
    squares that must not be free to trade ``k1`` against a tilt.  Three rounds
    are enough for the pair to settle.

    ``directions`` are the ENU unit vectors of the pinned stars, in the order the
    pins were placed; nothing here reads the overlay's own positions, which is
    what makes the answer independent of the drag.

    ``start`` is the camera the fit starts from -- the session passes the drag's
    camera (:meth:`AlignmentSession.camera_for`) or the previous pin fit -- and
    its attitude, mirror and radial curve are kept unless the pins say otherwise.
    ``coefficients=0`` fits the axis and the attitude only, which is how a
    four-pin fit that turns out to be unsolvable still leaves the chart useful.

    The radial fit is bounded (``method="trf"``; Levenberg-Marquardt cannot take
    bounds): the centre stays within :data:`PIN_CENTRE_LIMIT` of the middle of
    the frame and each ``k`` within +-1.  Unbounded, four pins trade centre
    against ``k1`` until the polynomial folds over and the rim comes out tens of
    degrees wrong -- which is exactly the failure mode a minimum-pin fit invites.

    ``optimize_focal`` is on by default and the reason is measurable: the pins
    are overwhelmingly measuring the radial *scale*, so a frozen focal length
    forces the first ``k`` to absorb the difference between the overlay's lens
    and the real one.  On the sample frame the fitted focal length lands at
    520 px/rad against a 180 deg overlay's 440, and a four-pin fit with ``f``
    frozen comes out at 21 px rms with a non-physical ``k1`` instead of 2 px with
    the right sign.  Freezing it is still useful when the focal length is known
    independently.
    """
    points = list(points)
    sky = np.asarray(directions, dtype=np.float64)
    if not points:
        raise ValueError("no stars pinned yet: click a circle, then the star in the photo")
    if sky.shape != (len(points), 3):
        raise ValueError(
            f"{len(points)} pins need {len(points)} sky directions (one 3-vector"
            f" each), got an array of shape {sky.shape}"
        )
    clicks = np.array([[point.x, point.y] for point in points], dtype=np.float64)
    camera = CameraModel(lens=ideal_lens) if start is None else start

    wanting = len(points) >= MIN_PINS_FOR_LENS
    count = pin_coefficient_count(len(points)) if wanting else 0
    if coefficients is not None:
        count = max(0, min(int(coefficients), MAX_PIN_COEFFICIENTS))
    measured = count > 0

    half = 0.5 * float(height)
    focal_ideal = equiangular_pixels_per_radian(ideal_lens, float(height))
    # The near-axis scale of the curve the fit starts from: with sigma = r/(H/2)
    # and theta = c1 sigma + ..., r = theta * (H/2) / c1 is what the pins would
    # give if the photograph agreed with that curve exactly.
    focal_start = half / float(camera.lens.coeffs_rad[0])

    notes: list[str] = []
    improper = False
    focal_current = focal_start
    coefficients_current: tuple[float, ...] = ()

    def predicted_pixels(centre_x, centre_y, focal, ks, rotation, lens=None):
        """Pixels the model puts the pinned stars at, for one attitude.

        The ``(f, k)`` form is the one the optimiser solves; ``lens`` selects the
        package's ``theta(sigma)`` series instead, which is what a fit below four
        pins has to use -- there is no ``(f, k)`` there to solve, so the curve the
        fit started from is used as it stands.  Writing the projection out here
        rather than building a :class:`~lens_calibration.model.LensModel` per
        evaluation is what keeps a click's answer under a frame or two.
        """
        in_camera = sky @ rotation
        theta = np.arccos(np.clip(in_camera[:, 2], -1.0, 1.0))
        phi = np.arctan2(in_camera[:, 1], in_camera[:, 0])
        if lens is None:
            series = np.ones_like(theta)
            square = theta * theta
            for order, k in enumerate(ks):
                series = series + float(k) * square ** (order + 1)
            radius = focal * theta * series
        else:
            radius = lens.sigma(theta) * half
        # y points *down* the raster, and a mirrored raster is flipped left to
        # right about the axis -- which is `plane_from_pixel` in two lines.
        hand = -1.0 if camera.mirrored else 1.0
        return centre_x + hand * radius * np.cos(phi), centre_y + radius * np.sin(phi)

    def turn_attitude() -> np.ndarray:
        """Closed-form attitude through the pins; returns the rotation it found.

        Kabsch over the clicked pixels against their catalogue stars, which is
        the best rotation there is for whatever lens and axis the camera
        currently has.  It is also the handedness test: an improper optimum means
        the raster is a mirror image of the sky.  That is *reported* (``improper``)
        rather than acted on here: three exact pairs can fit either way round, so
        the caller weighs the two hands against each other before turning
        anybody's photograph over.
        """
        nonlocal improper
        if len(points) < MIN_PINS_FOR_ATTITUDE:
            return camera.rotation()
        vectors = camera.camera_vectors(clicks[:, 0], clicks[:, 1], float(height))
        rotation, found = rotation_from_pairs(vectors, sky)
        if found and len(points) >= MIN_PINS_FOR_MIRROR:
            improper = True
        return rotation

    def shift_axis() -> None:
        """Move the optical axis so the pins' residuals average out.

        Shifting the centre shifts every prediction with it, one for one, so the
        least-squares answer is the mean residual and there is nothing to
        iterate.  This is the whole of a one-pin fit: two numbers from two
        equations, answered by the first click.
        """
        nonlocal camera
        predicted_x, predicted_y = camera.pixel_from_enu(sky, float(height))
        limit = PIN_CENTRE_LIMIT * min(float(width), float(height))
        centre_x = float(camera.xc) * float(height) + float(np.mean(clicks[:, 0] - predicted_x))
        centre_y = float(camera.yc) * float(height) + float(np.mean(clicks[:, 1] - predicted_y))
        centre_x = min(max(centre_x, 0.5 * float(width) - limit), 0.5 * float(width) + limit)
        centre_y = min(max(centre_y, 0.5 * float(height) - limit), 0.5 * float(height) + limit)
        camera = camera.with_updates(xc=centre_x / float(height), yc=centre_y / float(height))

    def joint_fit(terms_count: int, lens_solved: bool) -> bool:
        """Least squares over the axis, the attitude and (maybe) the lens.

        One solve rather than an alternation of attitude and radius, because the
        two are coupled in a way that alternation handles badly: Kabsch reads a
        *radius* error as a tilt (a tilt stretches one side of the frame and
        shrinks the other, and so does the wrong scale on a ring of pins), so
        each pass hands the next one a rotation that is compensating for the
        other's error and the pair can stall a long way from the truth.  Starting
        from Kabsch and letting everything move together is what avoids that.

        The attitude is parameterised as a small rotation vector *added* to the
        Kabsch answer, so the optimiser never sees the ``tan`` singularity of the
        (roll, tilt) pair and a bad pin cannot turn into a camera looking at the
        ground.

        Returns ``False`` when the solved curve cannot be written as a lens at
        all, which is the caller's cue to drop a term: a curve fitted past its
        evidence folds over just short of the rim, and a rim that cannot be
        inverted is worse than one term less.
        """
        nonlocal camera, focal_current, coefficients_current
        axes = 3 if len(points) >= MIN_PINS_FOR_MIRROR else 1
        if axes == 3:
            rotation_start = turn_attitude()
        else:
            # Two pins are four equations; Kabsch would spend three of them on a
            # rotation that can only look like a tilt because the axis has not
            # been solved yet.  Start from the attitude the camera already has and
            # let the roll and the axis settle together instead.
            rotation_start = camera.rotation()
        limit = PIN_CENTRE_LIMIT * min(float(width), float(height))
        solve_focal = lens_solved and optimize_focal

        def residuals(params: np.ndarray) -> np.ndarray:
            index = 2
            focal = focal_current
            if solve_focal:
                focal = float(params[index])
                index += 1
            ks = params[index : index + terms_count]
            index += terms_count
            step = np.zeros(3)
            step[3 - axes :] = params[index : index + axes]
            rotation = rotation_start @ cv2.Rodrigues(step)[0]
            predicted_x, predicted_y = predicted_pixels(
                float(params[0]), float(params[1]), focal, ks, rotation,
                None if lens_solved else lens_start,
            )
            return np.concatenate([predicted_x - clicks[:, 0], predicted_y - clicks[:, 1]])

        lower = [0.5 * float(width) - limit, 0.5 * float(height) - limit]
        upper = [0.5 * float(width) + limit, 0.5 * float(height) + limit]
        start = [
            float(np.clip(camera.xc * float(height), lower[0], upper[0])),
            float(np.clip(camera.yc * float(height), lower[1], upper[1])),
        ]
        if solve_focal:
            lower.append(0.2 * focal_ideal)
            upper.append(5.0 * focal_ideal)
            start.append(float(np.clip(focal_current, 0.2 * focal_ideal, 5.0 * focal_ideal)))
        lower += [-1.0] * terms_count
        upper += [1.0] * terms_count
        start += [0.0] * terms_count
        lower += [-PIN_ATTITUDE_STEP] * axes
        upper += [PIN_ATTITUDE_STEP] * axes
        start += [0.0] * axes
        solved = least_squares(
            residuals,
            np.array(start, dtype=np.float64),
            bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
            method="trf",
        )
        index = 2
        focal_solved = focal_current
        if solve_focal:
            focal_solved = float(solved.x[index])
            index += 1
        coefficients_solved = tuple(float(k) for k in solved.x[index : index + terms_count])
        index += terms_count
        step = np.zeros(3)
        step[3 - axes :] = solved.x[index : index + axes]
        rotation = rotation_start @ cv2.Rodrigues(step)[0]
        if rotation[2, 2] < math.cos(math.radians(PIN_TILT_LIMIT_DEG)):
            pointing = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
            raise ValueError(
                f"the pins put the optical axis {pointing:.0f} deg from the zenith, which"
                f" is past the {PIN_TILT_LIMIT_DEG:.0f} deg this attitude description can"
                " write down: is a pin on the wrong star?"
            )
        roll, tilt_x, tilt_y = attitude_from_rotation(rotation)
        camera = camera.with_updates(
            roll=roll, tilt_x=tilt_x, tilt_y=tilt_y,
            xc=float(solved.x[0]) / float(height),
            yc=float(solved.x[1]) / float(height),
        )
        if not lens_solved:
            return True
        try:
            lens = LensModel.from_fisheye(
                focal_solved,
                coefficients_solved,
                height_px=float(height),
                sigma_edge=ideal_lens.sigma_edge,
            )
        except ValueError:
            return False
        if not lens.monotonic():
            # The restated curve turns back on itself somewhere inside the disc:
            # past that point a radius is no field angle at all, and every
            # projection through the lens would be a lie.  One term less, then.
            return False
        focal_current = focal_solved
        coefficients_current = coefficients_solved
        camera = camera.with_updates(lens=lens)
        return True

    lens_start = camera.lens
    if len(points) < MIN_PINS_FOR_ATTITUDE:
        # One pin is two equations: the axis moves onto the star and nothing else
        # can be claimed.  It is also the stage that answers a click instantly.
        shift_axis()
        terms = 0
    else:
        terms = count
        while not joint_fit(terms, measured):
            if terms == 0:
                raise ValueError(
                    "the pinned stars do not describe a physical lens -- check the"
                    " pins, or pin another star between them"
                )
            terms -= 1

    # Residuals, computed through the *stored* model rather than through the
    # ``(f, k)`` form the optimiser worked in: the page draws the ring where the
    # converted model puts it, and a line whose length disagreed with the
    # reported number would be a lie about the fit.  The conversion costs a
    # fraction of a pixel here and up to a couple at the rim, and charging it to
    # the pins is what keeps the two honest.
    predicted_x, predicted_y = camera.pixel_from_enu(sky, float(height))
    predicted = np.column_stack([predicted_x, predicted_y])
    residual = np.hypot(predicted[:, 0] - clicks[:, 0], predicted[:, 1] - clicks[:, 1])
    clicked = camera.enu_from_pixel(clicks[:, 0], clicks[:, 1], float(height))
    cosine = np.clip(np.sum(clicked * sky, axis=1), -1.0, 1.0)
    angle_error = np.degrees(np.arccos(cosine))
    covers = float(
        np.max(
            np.hypot(
                clicks[:, 0] - float(camera.xc) * float(height),
                clicks[:, 1] - float(camera.yc) * float(height),
            )
        )
        / half
    )
    restate = 0.0
    if measured:
        # Over the stretch the pins actually measured: past a fold, or past the
        # rim, the series is an extrapolation and its disagreement with the
        # (f, k) curve says more about the extrapolation than about the fit.
        try:
            edge = min(float(camera.lens.sigma_edge) * half, fisheye_fold_radius(
                focal_current, coefficients_current
            ))
        except ValueError:
            edge = float(camera.lens.sigma_edge) * half
        radius = np.linspace(0.0, edge, 200)
        theta = camera.lens.theta(radius / half)
        restate = float(
            np.max(
                np.abs(
                    fisheye_radius(theta, focal_current, coefficients_current) - radius
                )
            )
        )

    if not measured:
        notes.append(
            f"{len(points)} pinned star{'s' if len(points) != 1 else ''} place the axis"
            f" and the attitude; the lens is still the seed -- {MIN_PINS_FOR_LENS}"
            " stars measure it"
        )
    else:
        if terms < count:
            notes.append(
                f"{len(points)} pins asked for {count} distortion terms, but {count}"
                f" folds the curve before the disc rim: fitted {terms} instead"
            )
        if terms == 1 and len(points) == MIN_PINS_FOR_LENS:
            notes.append(
                f"{len(points)} stars determine the centre and the theta^3 term only:"
                " pin a fifth to solve the theta^5 term as well"
            )
        elif terms == MAX_PIN_COEFFICIENTS and len(points) > MIN_PINS_FOR_LENS + terms - 1:
            notes.append(
                f"{len(points)} pins, but the fit stops at the theta^{MAX_PIN_ORDER_DEG}"
                f" term: stars past the first {MIN_PINS_FOR_LENS + terms - 1} over-determine"
                " the same unknowns rather than adding coefficients"
            )
        try:
            fold = fisheye_fold_radius(focal_current, coefficients_current)
        except ValueError:
            fold = float("inf")
        disc = float(camera.lens.sigma_edge) * half
        if fold < disc:
            notes.append(
                f"the fitted curve folds over at {fold:.0f} px, inside the {disc:.0f} px"
                " disc: the rim past it is extrapolated, so pin a star out there"
            )
    if covers < PIN_COVERAGE_WARN_SIGMA:
        notes.append(
            f"every pin is inside sigma {covers:.2f}: the rim is extrapolated,"
            " pin a star near the edge of the disc"
        )
    if not camera.lens.monotonic():
        notes.append("the fitted curve is not monotone across the disc")
    # Two equations per pin against the unknowns the fit was given: one spare
    # equation states a residual but cannot separate the tilt from a distortion
    # coefficient, and a tilt is very good at imitating a radius error on one side
    # of the frame.  Say so, rather than letting a minimum-pin fit report a
    # pointing it could not have measured.
    unknowns = 2 + (1 if (measured and optimize_focal) else 0) + terms + (
        3 if len(points) >= MIN_PINS_FOR_MIRROR else 1
    )
    if measured and 2 * len(points) - unknowns <= 1:
        notes.append(
            f"{len(points)} pins for {unknowns} unknowns: the tilt and k1 can trade"
            " against each other, so pin more stars before trusting the pointing"
        )

    return PinnedCamera(
        camera=camera,
        measured=bool(measured),
        improper=bool(improper),
        focal_px=float(focal_current) if measured else None,
        coefficients=tuple(float(k) for k in coefficients_current) if measured else (),
        rms_px=float(np.sqrt(np.mean(np.square(residual)))),
        max_px=float(np.max(residual)),
        rms_deg=float(np.sqrt(np.mean(np.square(angle_error)))),
        covers_sigma=covers,
        restate_px=restate,
        notes=tuple(notes),
        points=tuple(points),
        predicted=tuple((float(x), float(y)) for x, y in predicted),
        residuals=tuple(float(value) for value in residual),
    )


#: EXIF tag id for the orientation flag: the browser honours it and the raster
#: (``cv2.imread``) does not, so a frame that carries one is re-encoded instead
#: of being handed over as it lies on disk.
TAG_ORIENTATION = 274


class AlignmentSession:
    """Everything known about one frame, loaded lazily where it is expensive."""

    def __init__(
        self,
        path: str | Path,
        meta: ExposureMeta,
        gray: np.ndarray,
        frame_image: bytes,
        circle: CircleFit,
        calibration_dir: str | Path | None = None,
        frame_mimetype: str = "image/jpeg",
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        self.path = str(path)
        self.meta = meta
        self.gray = gray
        #: The frame as the page receives it: the file's own pixels, full size, no
        #: tone curve (see :func:`_frame_bytes`).  The raster in ``gray`` is what
        #: the fit works in, and the two are the same picture at two sizes.
        self.frame_image = frame_image
        self.frame_mimetype = frame_mimetype
        self.frame_size = (
            (int(gray.shape[1]), int(gray.shape[0]))
            if frame_size is None
            else (int(frame_size[0]), int(frame_size[1]))
        )
        self.circle = circle

        self.calibration_dir = None if calibration_dir is None else Path(calibration_dir)
        self.utc_override: dt.datetime | None = None
        self.latitude_override: float | None = None
        self.longitude_override: float | None = None
        #: The camera the user's pins measured, if any.  It is a property of the
        #: rig rather than of the sky, so it survives a change of time/place and
        #: is only replaced (or dropped) by the next pin refit.  ``measured``
        #: says whether the *lens* came with it: one pin already places the axis.
        self.pins: PinnedCamera | None = None
        self._field: StarField | None = None
        #: Last corrected preview: ``(key, png bytes)``, where the key is the
        #: camera it was drawn from.  Rendering one is a full-resolution resample
        #: and a PNG to go with it, so it is kept until the camera moves.
        self._corrected: tuple[str, bytes] | None = None

    # --------------------------------------------------------------- loading
    @classmethod
    def open(
        cls,
        path: str | Path,
        work_height: int = WORK_HEIGHT,
        calibration_dir: str | Path | None = None,
    ) -> "AlignmentSession":
        """Read one frame: EXIF, the work raster, the image circle, the bytes shown.

        ``calibration_dir=None`` saves the artefact beside the photograph (see
        :meth:`calibration_path`); a directory puts it there instead.
        """
        meta = read_exposure_meta(str(path))
        frame = _read_frame(Path(path))
        if frame is None:
            raise FileNotFoundError(f"cannot read {path}")
        scale = float(work_height) / float(frame.shape[0])
        size = (max(8, int(round(frame.shape[1] * scale))), int(round(work_height)))
        raster = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)
        circle = fit_image_circle(gray, work_height=0)
        image, mimetype = _frame_bytes(Path(path), frame)
        LOGGER.info(
            "%s: %dx%d frame shown, %dx%d work raster, disc centre %.4f/%.4f, sigma_edge %.3f",
            Path(path).name, frame.shape[1], frame.shape[0],
            raster.shape[1], raster.shape[0],
            circle.xc, circle.yc, circle.sigma_edge,
        )
        return cls(
            path, meta, gray, image, circle, calibration_dir,
            frame_mimetype=mimetype,
            frame_size=(frame.shape[1], frame.shape[0]),
        )

    # ------------------------------------------------------------ frame shape
    @property
    def height(self) -> float:
        """Height of the work raster in pixels."""
        return float(self.gray.shape[0])

    @property
    def width(self) -> float:
        """Width of the work raster in pixels."""
        return float(self.gray.shape[1])

    @property
    def frame_width(self) -> int:
        """Width of the photograph itself, in its own pixels."""
        return int(self.frame_size[0])

    @property
    def frame_height(self) -> int:
        """Height of the photograph itself, in its own pixels."""
        return int(self.frame_size[1])

    @property
    def stem(self) -> str:
        """File name of the frame without its extension."""
        return Path(self.path).stem

    # -------------------------------------------------------- time and place
    @property
    def utc(self) -> dt.datetime | None:
        """Capture instant actually used (EXIF unless overridden)."""
        return self.utc_override or self.meta.captured_utc

    @property
    def latitude(self) -> float | None:
        """Observer latitude actually used."""
        if self.latitude_override is not None:
            return self.latitude_override
        return self.meta.latitude

    @property
    def longitude(self) -> float | None:
        """Observer longitude actually used."""
        if self.longitude_override is not None:
            return self.longitude_override
        return self.meta.longitude

    def configure(
        self,
        utc: dt.datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> "AlignmentSession":
        """Override the time / place (``None`` keeps the current value).

        EXIF is often wrong in exactly the way that matters here: Sony writes
        the camera clock, which is normally local time, and without an offset
        tag the file looks like UTC.  A wrong hour rotates the whole sky, so the
        UI lets the user correct it and the field is then rebuilt.
        """
        changed = False
        if utc is not None:
            self.utc_override = utc if utc.tzinfo else utc.replace(tzinfo=dt.timezone.utc)
            changed = True
        if latitude is not None:
            self.latitude_override = float(latitude)
            changed = True
        if longitude is not None:
            self.longitude_override = float(longitude)
            changed = True
        if changed:
            self._field = None
        return self

    # -------------------------------------------------------------- lazy bits
    @property
    def field(self) -> StarField:
        """Catalogue stars above the horizon at the capture instant."""
        if self._field is None:
            utc = self.utc
            if utc is None:
                raise ValueError("no capture time: enter one in the time panel")
            if self.latitude is None or self.longitude is None:
                raise ValueError("no observer position: enter latitude and longitude")
            self._field = star_field(
                utc, self.latitude, self.longitude,
                elevation_m=self.meta.altitude_m or 0.0,
            )
        return self._field

    # --------------------------------------------------------------- overlay
    def seed_focal_px(self, focal_px: float | None = None) -> float:
        """Focal length of the overlay's fisheye, in work-raster pixels per radian.

        :data:`SEED_FOCAL_PX` is quoted for :data:`WORK_HEIGHT`; a lens model is
        dimensionless, so the same lens at another raster height has a focal
        length in proportion.  An explicit value from the page is taken as given
        -- it is in pixels of the raster the page is looking at.
        """
        if focal_px is None:
            value = SEED_FOCAL_PX * self.height / float(WORK_HEIGHT)
        else:
            value = float(focal_px)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"the focal length must be positive pixels per radian, got {value!r}")
        return value

    def seed_coefficients(self, coefficients: Sequence[float] | None = None) -> tuple[float, ...]:
        """``k`` terms of the overlay's fisheye (``k1 .. k4``, ``cv2.fisheye``)."""
        if coefficients is None:
            return tuple(float(k) for k in SEED_COEFFICIENTS)
        values = tuple(float(k) for k in coefficients)
        if len(values) > MAX_COEFFICIENTS:
            raise ValueError(
                f"at most {MAX_COEFFICIENTS} k terms, got {len(values)}"
            )
        if not all(math.isfinite(k) for k in values):
            raise ValueError(f"the k terms must be finite numbers, got {values!r}")
        return values

    def ideal_camera(
        self,
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> CameraModel:
        """Zenith-pointing camera the overlay is drawn with.

        ``fisheye`` -- the default -- is the measured lens itself: the seed
        focal length, ``k`` terms and optical centre (:data:`SEED_FOCAL_PX`,
        :data:`SEED_COEFFICIENTS`, :data:`SEED_CENTRE_PX`), or whatever the page
        says the rig measures.  The analytic families stay available for
        comparison, but there is no reason to align against a projection the lens
        does not make.

        The measured camera's centre is the rig's, which is not quite the middle
        of the frame; the families have no rig behind them and stay centred.  The
        disc *radius* is scaled to the measured image circle, so the overlay rim
        lands on the real rim and the user can drag the raster onto it.
        """
        if projection == SEED_PROJECTION:
            lens = LensModel.from_fisheye(
                self.seed_focal_px(focal_px),
                self.seed_coefficients(coefficients),
                height_px=self.height,
                sigma_edge=self.circle.sigma_edge,
            )
            # SEED_CENTRE_PX is quoted in pixels of a WORK_HEIGHT raster, and
            # xc/yc are fractions of the frame height, so the ratio is invariant.
            xc, yc = (SEED_CENTRE_PX[0] / WORK_HEIGHT, SEED_CENTRE_PX[1] / WORK_HEIGHT)
        elif projection in FAMILIES:
            lens = LensModel.from_family(
                projection, fov_deg=float(fov_deg), sigma_edge=self.circle.sigma_edge
            )
            xc = yc = 0.5
        else:
            raise ValueError(
                f"unknown projection {projection!r}, expected one of {PROJECTIONS}"
            )
        return CameraModel(lens=lens, xc=xc, yc=yc)

    def seed_lens(self) -> LensModel:
        """The rig's lens before any pins: the overlay's own radial model."""
        return self.ideal_camera(SEED_PROJECTION).lens

    def overlay(
        self,
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
        camera: CameraModel | None = None,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
        alignment: Alignment | None = None,
    ) -> dict:
        """The sky as this frame's camera sees it, drawn in screen pixels.

        ``camera`` is the measured one (:class:`PinnedCamera`); without it the
        overlay is the sky through the camera the user's drag implies
        (:meth:`camera_for`), which reproduces the old "drag the photo under the
        chart" behaviour exactly -- a rotated, shifted *chart*, not a rotated,
        shifted sky.

        Every position here is a *screen* position: projected through the camera
        into pixels of the photograph, then through ``alignment`` into the pixels
        the page draws in.  That composition is the whole of the registration --
        the rings sit on the photo's stars because the camera says where in the
        photo they are, and they follow the photo when it is dragged because the
        last step is the transform the photo itself is drawn with.  Once a
        calibration exists, dragging changes the view and nothing else, which is
        why the fit cannot be argued with by moving the frame around.

        The star selection follows the camera too: zenith distance from *its*
        optical axis, so a frame that does not point at the zenith is offered the
        stars it actually photographed.
        """
        view = Alignment() if alignment is None else alignment
        ideal = self.ideal_camera(projection, fov_deg, focal_px, coefficients)
        if camera is not None:
            placed = camera
        else:
            placed = self.camera_for(
                view, projection, fov_deg, focal_px=focal_px, coefficients=coefficients
            )
        stars = self.field
        axis = placed.rotation()[:, 2]
        theta = np.arccos(np.clip(stars.enu @ axis, -1.0, 1.0))
        keep = (stars.vmag <= float(mag_limit)) & (theta <= placed.lens.theta_edge)
        alt = stars.alt[keep]
        az = stars.az[keep]
        raster_x, raster_y = placed.pixel_from_altaz(alt, az, self.height)
        x, y = view.forward(raster_x, raster_y, self.width, self.height)
        rings = [
            {
                "x": float(px),
                "y": float(py),
                "name": name,
                "vmag": float(mag),
                "alt_deg": float(np.degrees(a)),
                "theta_deg": float(np.degrees(t)),
            }
            for px, py, name, mag, a, t in zip(
                np.atleast_1d(x), np.atleast_1d(y),
                _take(stars.names, keep), stars.vmag[keep], alt, theta[keep],
            )
        ]
        rings.sort(key=lambda ring: ring["vmag"])
        axis_x, axis_y = placed.centre_px(self.height)
        centre_x, centre_y = view.forward(axis_x, axis_y, self.width, self.height)
        # Where the zenith lands: with the axis tilted this is not the optical
        # centre any more, and the cross the page draws marks the sky, not the lens.
        zenith_x, zenith_y = view.forward(
            *placed.pixel_from_enu(np.array([[0.0, 0.0, 1.0]]), self.height),
            self.width,
            self.height,
        )
        measured = projection == SEED_PROJECTION
        return {
            "projection": projection,
            "fov_deg": float(2.0 * placed.lens.theta_edge_deg if measured else fov_deg),
            "focal_px": float(self.seed_focal_px(focal_px)) if measured else None,
            "coefficients": list(self.seed_coefficients(coefficients)) if measured else None,
            "calibrated": camera is not None,
            "theta_edge_deg": float(placed.lens.theta_edge_deg),
            "radius_px": float(placed.lens.sigma_edge * 0.5 * self.height),
            "centre": {
                "x": float(np.atleast_1d(centre_x)[0]),
                "y": float(np.atleast_1d(centre_y)[0]),
            },
            "zenith": {
                "x": float(np.atleast_1d(zenith_x)[0]),
                "y": float(np.atleast_1d(zenith_y)[0]),
            },
            "mag_limit": float(mag_limit),
            "count": len(rings),
            "grid": self._grid_layer(placed, view),
            "stars": rings,
            "aids": self._aid_layer(placed, view),
        }
    def _grid_layer(self, camera: CameraModel, view: Alignment) -> dict:
        """Alt-az grid: rings at fixed zenith distance and the four meridians.

        A fixed zenith distance is a circle *on the sky*, and a camera tilted away
        from the zenith does not photograph it as a circle -- the frame is
        compressed on the side the axis leans towards.  The rings are therefore
        projected sample by sample and drawn as polylines (the page used to
        stroke arcs from a radius, which is the same thing only while the axis
        points at the zenith), and the stretch of a curve that leaves the image
        circle is dropped rather than drawn along a rim that is not there.

        The curves come from the camera being drawn, so they stay exact for any
        projection and field of view: only the page's scale step is left.
        """
        edge = float(camera.lens.theta_edge)
        azimuth = np.radians(np.linspace(0.0, 360.0, GRID_SAMPLES, endpoint=False))
        rings = []
        for theta_deg in GRID_THETA_DEG:
            theta = math.radians(theta_deg)
            if theta > edge:
                continue
            directions = enu_from_altaz(
                np.full_like(azimuth, 0.5 * math.pi - theta), azimuth
            )
            points = self._curve(camera, view, directions, edge)
            if len(points) >= 2:
                rings.append({"theta_deg": float(theta_deg), "points": points})
        meridians = []
        altitude = np.linspace(0.0, 0.5 * math.pi, max(8, GRID_SAMPLES // 2))
        for az_deg, label in ((0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W")):
            directions = enu_from_altaz(
                altitude, np.full_like(altitude, math.radians(az_deg))
            )
            points = self._curve(camera, view, directions, edge)
            if len(points) < 2:
                continue
            # The label goes where the meridian crosses the horizon, which is the
            # first sample of the curve: the samples run outwards from the zenith.
            meridians.append(
                {
                    "az_deg": float(az_deg),
                    "label": label,
                    "points": points,
                    "label_x": points[0][0],
                    "label_y": points[0][1],
                }
            )
        return {"rings": rings, "meridians": meridians}

    def _curve(
        self, camera: CameraModel, view: Alignment, directions: np.ndarray, edge: float
    ) -> list[list[float]]:
        """One screen-space polyline: ``directions`` projected, inside the disc.

        Only the longest run of samples inside the image circle is kept.  A
        tilted camera can, right at the rim, cut a ring into two arcs; drawing the
        longer one and dropping the other is the honest choice, because a line
        joining them would invent sky the frame never saw.
        """
        in_camera = np.asarray(directions, dtype=np.float64) @ camera.rotation()
        inside = np.arccos(np.clip(in_camera[:, 2], -1.0, 1.0)) <= edge
        count = len(inside)
        if not inside.any():
            return []
        if inside.all():
            # Closed: repeat the first sample so the page can stroke one path.
            run = np.arange(count + 1) % count
        else:
            # Start the scan at a gap, so a run that wraps round 0/360 is one run.
            offset = int(np.argmax(~inside))
            rolled = np.roll(inside, -offset)
            # int8, because np.diff on a boolean array is a bool and `== -1` then
            # never matches: the runs would come out as empty pairs.
            flags = np.concatenate([[False], rolled, [False]]).astype(np.int8)
            edges = np.diff(flags)
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            if not len(starts) or not len(stops):
                return []
            best = int(np.argmax(stops - starts))
            run = (offset + np.arange(starts[best], stops[best])) % count
        raster_x, raster_y = camera.pixel_from_enu(directions[run], self.height)
        screen_x, screen_y = view.forward(raster_x, raster_y, self.width, self.height)
        return [
            [round(float(px), 2), round(float(py), 2)]
            for px, py in zip(np.atleast_1d(screen_x), np.atleast_1d(screen_y))
        ]

    def _aid_layer(self, camera: CameraModel, view: Alignment = Alignment()) -> dict:
        """Clusters, nebulae and the Milky Way, projected for orientation only.

        These answer the question the rings cannot -- "is that smudge the
        Pleiades or the Double Cluster?" -- while the photo is still being
        dragged.  Nothing here can be fitted: a pin has to hang off a catalogue
        star, whose position is a point, whereas a cluster is a fuzzy patch with
        no centre worth clicking, so the layer is reported *beside* ``stars`` and
        the page's hit test never sees it.

        Like the grid, both the band and the markers are projected through the
        camera being drawn, so they follow a calibration instead of drifting off
        it.  A band sample whose edges leave the disc is dropped, and each run of
        surviving samples becomes one filled polygon -- where the band exits the
        field the polygon closes across its own width, which is what the exit
        should look like, so no clipping maths is needed.
        """
        empty: dict = {"objects": [], "band": []}
        if self.utc is None or self.latitude is None or self.longitude is None:
            return empty
        sky = sky_aids(self.utc, self.latitude, self.longitude, self.meta.altitude_m or 0.0)
        if not len(sky):
            return empty
        axis = camera.rotation()[:, 2]
        theta_edge = float(camera.lens.theta_edge)

        theta = np.arccos(np.clip(enu_from_altaz(sky.alt, sky.az) @ axis, -1.0, 1.0))
        keep = theta <= theta_edge
        x, y = camera.pixel_from_altaz(sky.alt[keep], sky.az[keep], self.height)
        x, y = view.forward(x, y, self.width, self.height)
        objects = [
            {
                "name": name,
                "kind": kind,
                "x": float(px),
                "y": float(py),
                "theta_deg": float(np.degrees(t)),
                "radius_deg": float(np.degrees(radius)),
                # Drawn at the object's angular radius, so a big one reads big.
                "radius_px": float(camera.lens.sigma(float(radius)) * 0.5 * self.height),
            }
            for name, kind, px, py, t, radius in zip(
                _take(sky.names, keep), _take(sky.kinds, keep),
                np.atleast_1d(x), np.atleast_1d(y), theta[keep], sky.radius_deg[keep],
            )
        ]

        edge_alt = sky.band_alt.reshape(-1)
        edge_az = sky.band_az.reshape(-1)
        band_theta = np.arccos(
            np.clip(enu_from_altaz(edge_alt, edge_az) @ axis, -1.0, 1.0)
        ).reshape(sky.band_alt.shape)
        inside = np.all(band_theta <= theta_edge, axis=0)
        band_x, band_y = camera.pixel_from_altaz(edge_alt, edge_az, self.height)
        band_x, band_y = view.forward(band_x, band_y, self.width, self.height)
        band_x = np.asarray(band_x).reshape(sky.band_alt.shape)
        band_y = np.asarray(band_y).reshape(sky.band_alt.shape)
        # Padded with False so a run that starts or ends at sample 0 is closed
        # the same way; the transitions then come in start/stop pairs.
        flags = np.concatenate([[False], inside, [False]]).astype(np.int8)
        breaks = np.flatnonzero(np.diff(flags))
        band = []
        for start, stop in zip(breaks[::2], breaks[1::2]):
            if stop - start < 3:
                continue
            ring = [
                [round(float(band_x[0, index]), 1), round(float(band_y[0, index]), 1)]
                for index in range(start, stop)
            ]
            ring += [
                [round(float(band_x[1, index]), 1), round(float(band_y[1, index]), 1)]
                for index in range(stop - 1, start - 1, -1)
            ]
            band.append(ring)
        return {"objects": objects, "band": band}

    def _overlay_directions(
        self, camera: CameraModel, rings: int = 4, spokes: int = 12
    ) -> np.ndarray:
        """ENU unit vectors spread over the overlay disc, plus the disc centre."""
        theta = np.linspace(0.0, camera.lens.theta_edge, rings + 1)[1:]
        phi = np.linspace(0.0, 2.0 * math.pi, spokes, endpoint=False)
        theta, phi = np.meshgrid(theta, phi, indexing="ij")
        directions = enu_from_altaz(0.5 * math.pi - theta.ravel(), phi.ravel())
        return np.vstack([np.array([[0.0, 0.0, 1.0]]), directions])

    def camera_for(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        lens: LensModel | None = None,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> CameraModel:
        """Camera model described by the user's manual alignment.

        This is the *prior*: before two stars are pinned there is no attitude
        measurement at all, and the drag is the user's statement about where the
        sky is.  The pin fit starts here and replaces it (see
        :func:`fit_camera_from_pins`), so a wrong drag costs a slightly longer
        search and never a wrong calibration.

        The alignment says how the raster sits under the sky, so the stars' sky
        positions (:meth:`Alignment.inverse`) are the raster positions they must
        have.  Finding the camera that predicts those positions is then a
        rigid-body fit, which :func:`~lens_calibration.match.update_attitude`
        solves in closed form -- including the pixel convention, cross-checked by
        its improper-rotation test instead of by hand-derived signs.

        The optical centre is the overlay camera's own, mapped through the
        alignment: it starts at the seed's axis, which is off the middle of the
        frame, and a camera centred on the middle would leave a residual there
        that is not attitude at all.

        ``lens`` is the radial model the fit is expressed in.  It defaults to the
        overlay's own lens, which makes the result exactly reproduce the manual
        transform; pass the lens the pins measured for the calibration itself.
        """
        ideal = self.ideal_camera(projection, fov_deg, focal_px, coefficients)
        directions = self._overlay_directions(ideal)
        overlay_x, overlay_y = ideal.pixel_from_enu(directions, self.height)
        x, y = alignment.inverse(overlay_x, overlay_y, self.width, self.height)
        axis_x, axis_y = ideal.centre_px(self.height)
        centre_x, centre_y = alignment.inverse(
            np.array([axis_x]), np.array([axis_y]), self.width, self.height,
        )
        candidate = CameraModel(
            lens=lens if lens is not None else ideal.lens,
            xc=float(centre_x[0]) / self.height,
            yc=float(centre_y[0]) / self.height,
            mirrored=alignment.mirrored,
        )
        camera, _ = update_attitude(
            candidate, _pixel_stars(x, y, self.height), _direction_field(directions),
            _identity_correspondence(directions.shape[0]), self.height,
        )
        return camera

    def alignment_error_px(
        self,
        camera: CameraModel,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> float:
        """Largest pixel error between ``camera`` and the manual alignment.

        Should be numerical noise (~1e-9 px); anything larger means the manual
        transform and the camera model disagree.
        """
        ideal = self.ideal_camera(projection, fov_deg, focal_px, coefficients)
        directions = self._overlay_directions(ideal)
        overlay_x, overlay_y = ideal.pixel_from_enu(directions, self.height)
        x, y = alignment.inverse(overlay_x, overlay_y, self.width, self.height)
        px, py = camera.pixel_from_enu(directions, self.height)
        return float(np.max(np.hypot(px - x, py - y)))

    # -------------------------------------------------------- live alignment
    def _pin_directions(self, points: Sequence[ControlPoint]) -> np.ndarray:
        """ENU direction of every pinned star, looked up by name.

        This is what makes a pin a measurement rather than a nudge: *which* star
        it is comes from the catalogue, not from where the overlay drew the
        circle, so the answer cannot drift with the overlay that produced it --
        and a refit after the chart has moved still measures the same star.
        """
        field = self.field
        by_name: dict[str, int] = {}
        for index, name in enumerate(field.names):
            by_name.setdefault(str(name), int(index))
        directions = []
        for point in points:
            index = by_name.get(point.name)
            if index is None:
                raise ValueError(
                    f"{point.name!r} is not in the star field for this frame's time and"
                    " place: pin a star the overlay shows"
                )
            directions.append(field.enu[index])
        return np.array(directions, dtype=np.float64)

    def live_alignment(
        self,
        pins: Sequence[dict],
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> dict:
        """Refit the camera through the pinned stars.

        Each pin is a star the page named and a pixel of the photograph the user
        clicked under it; every pair is refitted at once, so a new pin corrects
        the camera as a whole rather than nudging it, and removing one leaves no
        trace of the fit that had it.  What comes out grows with the number of
        pins (:func:`fit_camera_from_pins`): the axis from one, the attitude from
        two, the two together from three, and from :data:`MIN_PINS_FOR_LENS` pins
        the lens as well.

        The answer carries the re-projected overlay every time, because every pin
        moves the camera, and the rings have to follow it.  The ``alignment`` it
        carries is usually the one that was sent, unchanged -- the drag is the
        user's and the fit does not touch it.  The one exception is the mirror:
        if the pins say the frame is a mirror image of the sky, the answer flips
        the flag, and the page shows the photo the same way round so the rings
        stay on the stars.

        A fit that cannot be solved with four pins must not throw away what two
        or three of them won, so the failure is reported beside a camera fitted
        without the lens instead of in place of it.
        """
        points: list[ControlPoint] = []
        for pin in pins:
            if not isinstance(pin, dict):
                raise ValueError("a control point must be an object")
            name = str(pin.get("name") or "").strip()
            if not name:
                raise ValueError("a control point needs the name of the star it pins")
            try:
                x, y = float(pin["x"]), float(pin["y"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"{name}: a control point needs x and y, in pixels of the photograph"
                ) from None
            points.append(ControlPoint(name, x, y))
        if not points:
            # An empty list is how the page forgets the pins: the camera goes back
            # to the reading of the sky the drag implies, which is what
            # /api/session draws, and the answer carries that overlay so the page
            # has one thing to draw either way.
            self.pins = None
            return {
                "alignment": alignment.to_dict(),
                "points": [],
                "rms_px": None,
                "max_px": None,
                "rms_deg": None,
                "camera": _camera_dict(
                    self.camera_for(
                        alignment, projection, fov_deg,
                        focal_px=focal_px, coefficients=coefficients,
                    ),
                    self.height,
                ),
                "lens": None,
                "lens_error": None,
                "overlay": self.overlay(
                    projection, fov_deg, mag_limit,
                    focal_px=focal_px, coefficients=coefficients, alignment=alignment,
                ),
            }
        directions = self._pin_directions(points)
        ideal = self.ideal_camera(projection, fov_deg, focal_px, coefficients)
        start = (
            self.pins.camera
            if self.pins is not None
            else self.camera_for(
                alignment, projection, fov_deg, focal_px=focal_px, coefficients=coefficients
            )
        )
        def fit_from(start_camera: CameraModel) -> tuple[PinnedCamera, str | None]:
            """Fit the camera; a lens that cannot be solved falls back to the axis.

            A four-pin lens fit can fail where a three-pin axis fit cannot, and
            that must not throw away what those pins won: the failure is reported
            beside a camera fitted without the lens instead of in place of it.
            """
            try:
                return fit_camera_from_pins(
                    points, directions, self.width, self.height, ideal.lens,
                    start=start_camera,
                ), None
            except ValueError as error:
                if len(points) < MIN_PINS_FOR_LENS:
                    raise
                LOGGER.info("%s: the pins do not calibrate the lens yet: %s", self.stem, error)
                return fit_camera_from_pins(
                    points, directions, self.width, self.height, ideal.lens,
                    start=start_camera, coefficients=0,
                ), str(error)

        pinned, lens_error = fit_from(start)
        if pinned.improper and len(points) >= MIN_PINS_FOR_MIRROR:
            # The pins describe a raster that is a mirror image of the sky.  Believe
            # them only when the other hand cannot explain the pins at all: three
            # exact pairs can fit either way round, and turning the user's
            # photograph over is not something to do on a coin toss.
            other, other_error = fit_from(start.with_updates(mirrored=not start.mirrored))
            if other.rms_px < 0.5 * pinned.rms_px:
                pinned, lens_error = other, other_error

        if pinned.camera.mirrored != alignment.mirrored:
            pinned = replace(pinned, notes=pinned.notes + (
                "the pins are a mirror image of the frame as it was drawn: the mirror"
                " was flipped to match, and the page turns the photo over with it",
            ))

        fitted = alignment.with_mirror(pinned.camera.mirrored)
        predicted = np.asarray(pinned.predicted, dtype=np.float64)
        drawn_x, drawn_y = fitted.forward(
            predicted[:, 0], predicted[:, 1], self.width, self.height
        )
        self.pins = pinned
        alt, az = altaz_from_enu(pinned.camera.rotation()[:, 2])
        LOGGER.info(
            "%s: %d pinned star(s): axis alt %.1f az %.1f, roll %.1f, %s, %.1f px rms",
            self.stem, len(points), float(np.degrees(alt)), float(np.degrees(az)),
            math.degrees(pinned.camera.roll),
            f"lens f {pinned.focal_px:.1f} px/rad" if pinned.measured else "seed lens",
            pinned.rms_px,
        )
        return {
            "alignment": fitted.to_dict(),
            "points": [
                point.to_dict(residual, (float(dx), float(dy)))
                for point, residual, dx, dy in zip(
                    pinned.points, pinned.residuals,
                    np.atleast_1d(drawn_x), np.atleast_1d(drawn_y),
                )
            ],
            "rms_px": pinned.rms_px,
            "max_px": pinned.max_px,
            "rms_deg": pinned.rms_deg,
            "camera": _camera_dict(pinned.camera, self.height),
            "lens": pinned.to_dict(self.height) if pinned.measured else None,
            "lens_error": lens_error,
            "overlay": self.overlay(
                projection,
                fov_deg,
                mag_limit,
                camera=pinned.camera,
                focal_px=focal_px,
                coefficients=coefficients,
                alignment=fitted,
            ),
        }

    def calibration_camera(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> CameraModel:
        """The camera the pins measured -- or, before them, the one the drag implies.

        With pins there is nothing to combine: the fit answered the whole
        question (lens, axis, attitude, mirror) in the photograph's own pixels,
        so the camera is the answer, and it does not depend on where the photo is
        drawn on screen.  Without pins the overlay's own lens stands in and the
        attitude comes from the drag (:meth:`camera_for`) -- that is the seed, and
        callers report it as such rather than pretending to a measurement.
        """
        if self.pins is not None:
            return self.pins.camera
        return self.camera_for(
            alignment, projection, fov_deg, focal_px=focal_px, coefficients=coefficients
        )

    # ------------------------------------------------------------- artefacts
    def calibration_payload(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> dict:
        """Calibration artefact readable by :func:`~lens_calibration.model.read_calibration`.

        ``lens_calibration/2``: the lens *and* the camera come from the pins --
        the optical centre, the attitude (roll and the tilt of the axis away from
        the zenith) and the mirror are all measured, in pixels of this frame.
        Version 1 carried a solver result instead, and there is no solver left to
        reproduce it -- ``lens_source`` says which of the two a file holds, because
        a file written before four stars were pinned holds the seed and not a
        calibration.

        ``alignment`` is the display transform the page happened to be using.  It
        is recorded for reproducing the screen, not because the camera depends on
        it: dragging the photo after it has been pinned cannot change the numbers
        above.
        """
        ideal = self.ideal_camera(projection, fov_deg, focal_px, coefficients)
        fisheye = projection == SEED_PROJECTION
        camera = self.calibration_camera(alignment, projection, fov_deg, focal_px, coefficients)
        measured = self.pins is not None and self.pins.measured
        notes = list(self.pins.notes) if self.pins is not None else [
            "no star was pinned, so the lens in this file is the seed, not a"
            " measurement: pin at least four stars before relying on it"
        ]
        return {
            "format": "lens_calibration/2",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image": {
                "path": self.path,
                "width": int(self.width),
                "height": int(self.height),
                "work_height": int(self.height),
            },
            "meta": self.meta.to_dict(),
            "time": {
                "captured_utc": self.utc.isoformat() if self.utc else None,
                "source": self.meta.time_source,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "position_source": self.meta.position_source,
            },
            "circle": self.circle.to_dict(),
            "lens": camera.lens.to_dict(),
            "lens_source": "pins" if measured else "seed",
            "camera": _camera_dict(camera, self.height),
            "overlay": {
                "projection": projection,
                "fov_deg": float(
                    2.0 * ideal.lens.theta_edge_deg if fisheye else fov_deg
                ),
                "focal_px": float(self.seed_focal_px(focal_px)) if fisheye else None,
                "coefficients": list(self.seed_coefficients(coefficients)) if fisheye else None,
            },
            "alignment": alignment.to_dict(),
            "pins": self.pins.to_dict(self.height) if self.pins is not None else None,
            "notes": notes,
        }

    def calibration_path(self) -> Path:
        """Where the artefact for this frame belongs: beside the photograph.

        The JSON describes *that file* -- its pixels, its clock, its place -- so
        it goes next to it, named after it, and a folder of frames carries its own
        calibrations with it.  ``calibration_dir`` still wins when it is set (the
        command line's ``--calibration-dir``), which is also what a browser upload
        needs: the copy the server reads is its own under the system temp
        directory, not the folder the user keeps the frame in.
        """
        if self.calibration_dir is None:
            source = Path(self.path)
            return source.with_name(f"{source.stem}.json")
        return self.calibration_dir / f"{self.stem}.json"

    def save_calibration(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> Path:
        """Write the calibration artefact and return its path."""
        path = self.calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.calibration_payload(
            alignment, projection, fov_deg, focal_px, coefficients
        )
        path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")
        LOGGER.info("wrote %s", path)
        return path

    # ------------------------------------------------------------- corrected
    def corrected_png(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
    ) -> bytes:
        """The photograph resampled onto an ideal equiangular lens, as PNG.

        Full resolution and the photograph's own size, so the page can lay it
        under the same overlay and swap between the two: a named star then stays
        with its circle only where the calibration says it should, which is the
        statement the eye checks in a flash comparison.  The *measured* camera is
        the one that says where the frame's pixels are -- it is what the resample
        inverts -- and the ideal camera through the same centre and the same rim
        is what the output follows
        (:meth:`~lens_calibration.geometry.CameraModel.ideal_equiangular`).

        Rendered when it is asked for (a 4000x4000 resample plus a PNG to match
        is not something to do on every page load) and kept until the camera it
        was drawn from changes.  Nothing is written to disk.
        """
        camera = self.calibration_camera(
            alignment, projection, fov_deg, focal_px, coefficients
        )
        ideal = camera.ideal_equiangular()
        key = json.dumps(
            [camera.to_dict(), ideal.to_dict(), self.path, self.frame_size],
            sort_keys=True,
        )
        if self._corrected is not None and self._corrected[0] == key:
            return self._corrected[1]
        image = _read_frame(Path(self.path))
        if (image.shape[1], image.shape[0]) != (self.frame_width, self.frame_height):
            raise ValueError(
                f"{Path(self.path).name} is now {image.shape[1]}x{image.shape[0]}, not the"
                f" {self.frame_width}x{self.frame_height} this session read: reopen the frame"
            )
        corrected = remap_to_ideal(image, camera, ideal)
        ok, buffer = cv2.imencode(".png", corrected)
        if not ok:
            raise ValueError(f"could not encode the corrected {Path(self.path).name} as PNG")
        data = buffer.tobytes()
        self._corrected = (key, data)
        LOGGER.info(
            "corrected preview of %s: %dx%d, %d bytes, %s",
            Path(self.path).name, self.frame_width, self.frame_height, len(data),
            "lens measured" if self.pins is not None and self.pins.measured else "seed lens",
        )
        return data

    # ---------------------------------------------------------------- summary
    def summary(
        self,
        alignment: Alignment = Alignment(),
        projection: str = SEED_PROJECTION,
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
        focal_px: float | None = None,
        coefficients: Sequence[float] | None = None,
        corrected: bool = False,
    ) -> dict:
        """Everything the page needs on load.

        ``corrected`` asks for the overlay to be drawn through the ideal
        equiangular lens the corrected preview follows, rather than through the
        measured one.  That is the whole reason it is a request flag: on the
        photograph the rings must mark where the *photograph* puts the stars, and
        on the corrected picture they must mark where the *ideal* lens does, or a
        flash comparison would show the calibration's own correction as an error.
        """
        overlay: dict | None = None
        error: str | None = None
        camera: dict | None = None
        check: float | None = None
        measured = self.pins is not None and self.pins.measured
        try:
            overlay_camera = self.pins.camera if self.pins is not None else None
            if corrected:
                overlay_camera = self.calibration_camera(
                    alignment, projection, fov_deg, focal_px, coefficients
                ).ideal_equiangular()
            overlay = self.overlay(
                projection, fov_deg, mag_limit,
                camera=overlay_camera,
                focal_px=focal_px,
                coefficients=coefficients,
                alignment=alignment,
            )
        except ValueError as exc:
            error = str(exc)
        if overlay is not None:
            # The camera the frame is understood with: the pins' answer once there
            # is one, and the drag's reading of the sky before that.
            camera = _camera_dict(
                self.calibration_camera(alignment, projection, fov_deg, focal_px, coefficients),
                self.height,
            )
            if self.pins is not None and self.pins.measured:
                # With a measured lens the interesting number is the price of
                # restating it: the fit solves (f, k...), the overlay draws with
                # the theta(sigma) series, and the rim is where they differ most.
                check = self.pins.restate_px
            else:
                # Without pins there is no lens to restate; what can be checked is
                # that the camera the drag implies really reproduces the drag.
                check = self.alignment_error_px(
                    self.camera_for(
                        alignment, projection, fov_deg, focal_px=focal_px, coefficients=coefficients
                    ),
                    alignment,
                    projection,
                    fov_deg,
                    focal_px,
                    coefficients,
                )

        warnings: list[str] = []
        if "ASSUMED" in self.meta.time_source:
            warnings.append(
                "the EXIF clock carries no timezone offset, so it is being read as UTC;"
                " a camera set to local time is wrong by the UTC offset"
            )
        # A seed lens described by two terms can fold over inside the disc the
        # frame measures: past the fold no radius corresponds to any angle, so
        # the rim -- and anything drawn near it -- is an extrapolation of the
        # series rather than a measurement.  Worth saying before the user pins
        # stars near the edge and wonders why the rings do not land.
        if projection == SEED_PROJECTION and not measured:
            try:
                fold = fisheye_fold_radius(
                    self.seed_focal_px(focal_px), self.seed_coefficients(coefficients)
                )
            except ValueError:
                fold = float("inf")
            disc = self.circle.sigma_edge * 0.5 * self.height
            if fold < disc:
                warnings.append(
                    f"the seed lens folds over at {fold:.0f} px, inside the {disc:.0f} px"
                    " disc: the rim is extrapolated past what the series can describe --"
                    " pin stars near the edge to measure the lens there"
                )
        return {
            "image": {
                "url": "/frame.jpg",
                "name": Path(self.path).name,
                # The raster the pins and the overlay are in, and the photograph
                # behind it: the page draws in the first and displays the second,
                # so it needs both to say "1:1" and to show the file's size.
                "width": int(self.width),
                "height": int(self.height),
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
            },
            "meta": self.meta.to_dict(),
            "time": {
                "captured_utc": self.utc.isoformat() if self.utc else None,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "source": self.meta.time_source,
                "position_source": self.meta.position_source,
            },
            "circle": self.circle.to_dict(),
            "overlay": overlay,
            "overlay_error": error,
            # The second picture the page can put under the overlay: the frame
            # resampled onto an ideal equiangular lens, drawn when the button is
            # pressed (see corrected_png).  Same size as the photograph, so the
            # swap is a flash comparison and nothing moves.
            "corrected": {
                "url": CORRECTED_URL,
                "active": bool(corrected),
                "projection": "equiangular",
            },
            "alignment": alignment.to_dict(),
            "camera": camera,
            "lens_source": "pins" if measured else "seed",
            "transform_check_px": check,
            "counts": {"catalogue": None if error else len(self.field)},
            "projections": list(PROJECTIONS),
            "warnings": warnings,
            "calibration_path": str(self.calibration_path()),
        }


# ------------------------------------------------------------------- helpers
def _take(values, index) -> list:
    """Select from a sequence with either a boolean mask or integer indices."""
    index = np.asarray(index)
    if index.dtype == bool:
        return [value for value, flag in zip(values, index) if flag]
    return [values[int(i)] for i in np.atleast_1d(index)]


def _pixel_stars(x, y, height: float) -> DetectedStars:
    """Detections built from bare pixel positions (for the attitude fit)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    zeros = np.zeros(x.shape, dtype=np.float64)
    return DetectedStars(
        x=x, y=y, flux=zeros, peak=zeros, fwhm=zeros, snr=zeros, height_px=float(height)
    )


def _direction_field(directions: np.ndarray) -> StarField:
    """A ``StarField`` of arbitrary sky directions (no catalogue involved)."""
    alt, az = altaz_from_enu(directions)
    count = directions.shape[0]
    return StarField(
        names=tuple(f"dir{index}" for index in range(count)),
        hip=np.arange(count),
        vmag=np.zeros(count),
        bv=np.zeros(count),
        alt=alt,
        az=az,
        enu=directions,
    )


def _identity_correspondence(count: int) -> Correspondence:
    """Pair every direction with itself (index ``i`` to index ``i``)."""
    index = np.arange(count)
    return Correspondence(det_index=index, star_index=index, sep_deg=np.zeros(count))


def _camera_dict(camera: CameraModel, height_px: float | None = None) -> dict:
    """JSON-ready camera description (centre in pixels of ``height_px``)."""
    height = float(height_px or WORK_HEIGHT)
    alt, az = altaz_from_enu(camera.rotation()[:, 2])
    return {
        "xc": float(camera.xc),
        "yc": float(camera.yc),
        "centre_px": {"x": float(camera.xc * height), "y": float(camera.yc * height)},
        "roll_deg": float(math.degrees(camera.roll)),
        "tilt_x_deg": float(math.degrees(camera.tilt_x)),
        "tilt_y_deg": float(math.degrees(camera.tilt_y)),
        # Where the optical axis points, which is the reading that says whether a
        # frame is a zenith frame at all: tilt is a tangent-plane offset and stops
        # meaning anything near the horizon.
        "pointing_alt_deg": float(math.degrees(alt)),
        "pointing_az_deg": float(math.degrees(az)),
        "mirrored": bool(camera.mirrored),
        "theta_edge_deg": float(camera.lens.theta_edge_deg),
        "sigma_edge": float(camera.lens.sigma_edge),
        "lens": camera.lens.to_dict(),
    }


def _plain(value):
    """Recursively convert numpy scalars, arrays and paths to JSON-friendly ones."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_frame(path: Path) -> np.ndarray:
    """The photograph as an array, read the way the page is served it.

    One reader for both uses: the work raster is a downscale of this array, and
    the corrected preview resamples it, so the two are the same picture at two
    sizes by construction rather than by two readers happening to agree (EXIF
    orientation included).
    """
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"cannot read {path}")
    return frame


def _frame_bytes(path: Path, frame: np.ndarray) -> tuple[bytes, str]:
    """The frame as the page receives it: the file's own pixels, untouched.

    Nothing here stretches, sharpens or resizes.  A night frame looks black until
    it is lifted, but the lift belongs to the *view*: the page has a gamma slider
    for it, it starts flat, and it can be taken back.  Baking one opinion into the
    bytes would settle that argument for the user, and downscaling them to the
    work raster would cap what the page can show -- the browser is given the whole
    photograph, so leaning in with the browser's own zoom reaches the stars the
    sensor recorded instead of the ones this server kept.

    A JPEG is handed over as it lies on disk: byte-identical, so no re-encode can
    soften a star the user is trying to centre.  Anything else -- or a JPEG that
    carries an EXIF orientation, which the raster ignores and the browser would
    honour -- is re-encoded losslessly as PNG from the array the geometry was
    fitted to, so the pixels on screen are the pixels the pins are measured
    against.
    """
    if path.suffix.lower() in (".jpg", ".jpeg") and _exif_orientation(path) in (None, 1):
        return path.read_bytes(), "image/jpeg"
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"could not encode {path.name} for display")
    return buffer.tobytes(), "image/png"


def _exif_orientation(path: Path) -> int | None:
    """EXIF orientation tag of a file, or ``None`` if it has none (or unreadable)."""
    try:
        with Image.open(path) as photo:
            return photo.getexif().get(TAG_ORIENTATION)
    except (AttributeError, KeyError, OSError, SyntaxError, TypeError, ValueError):
        # A format PIL cannot read is simply a file that says nothing about which
        # way is up; the raster is not rotated either, so the two agree.
        return None


