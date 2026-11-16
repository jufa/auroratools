"""Per-image state shared by the solver and the interactive web UI.

One :class:`AlignmentSession` owns everything that is expensive or stateful
about a single exposure -- the EXIF metadata, the downscaled work raster, the
image-circle fit, the star detections and the catalogue field -- so a web
request never has to reload the frame.

The interactive flow is deliberately *manual first*.  The overlay is an ideal
fisheye projection of the sky (equiangular, 180 deg across the disc) with the
bright catalogue stars ringed; the user rotates, mirrors and drags the raster
underneath it until the real stars sit in the rings.  :class:`Alignment` is
that manual transform and :meth:`AlignmentSession.camera_for` converts it into
the :class:`~lens_calibration.geometry.CameraModel` the solver starts from --
so the user's alignment *is* the initial guess, and nothing is solved until the
user asks for it.

Why the conversion is done numerically
-------------------------------------
A mirror/rotation/translation of the raster is a 2-D orthogonal map composed
with the ideal camera's radial map, so it is exactly representable by another
:class:`CameraModel` -- but only if the sign conventions of the web transform,
the pixel convention and the roll are all right.  Rather than derive those
signs (the historical source of the "everything solves mirrored" bug), the
transform is applied to a ring of known sky directions and the attitude is
recovered with the same closed-form Kabsch update the solver uses.  Mirroring
is cross-checked by the improper-rotation test inside
:func:`~lens_calibration.match.update_attitude`.

Pinning stars instead of dragging
---------------------------------
Dragging lines the whole sky up by eye; it cannot say *which* star the user is
looking at.  The page therefore also lets the user pin a star: click its circle
in the overlay, then click the same star in the photograph, and
:func:`fit_alignment` refits the entire transform through every pair at once --
one pin determines a shift, two a rotation as well, and any further pin is
least squares, so the leftover error of each pair is visible as a line.  That is
ordinary georeferencing, and it turns the manual step into a control-point
problem the solver can start from.

The fourth pin is where that stops being enough.  A rigid fit has three degrees
of freedom, so a second pin already exhausts it; the third can only *report* the
error that is left, and what is left is radial -- the ideal overlay and the real
lens disagree about where the rim is.  From :data:`MIN_PINS_FOR_LENS` pins on,
:func:`fit_lens_from_pins` therefore solves the radial model through the same
pairs: the optical centre, the focal length and the odd ``k`` coefficients of the
fisheye curve ``cv2.fisheye`` calibrates -- four pins buy ``k1``, the fifth
``k2``, and there the polynomial order stops
(:data:`MAX_PIN_ORDER_DEG`), so every further star is another equation for the
same five unknowns rather than another coefficient.  The page then draws the
circles through the lens that was measured, and the solver starts from it
(:meth:`AlignmentSession.seed_lens`) instead of from a nominal fisheye.

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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .aids import sky_aids
from .catalog import StarField, star_field
from .exifmeta import ExposureMeta, read_exposure_meta
from .geometry import CameraModel, altaz_from_enu, enu_from_altaz
from .match import (
    AttitudeSolution,
    CircleFit,
    Correspondence,
    DetectedStars,
    detect_sources,
    fit_image_circle,
    solve_attitude,
    update_attitude,
)
from .model import FAMILIES, LensModel, equiangular_pixels_per_radian

LOGGER = logging.getLogger(__name__)

#: Height of the raster the UI displays and the solver analyses.  The lens model
#: is dimensionless, so a calibration fitted here transfers to any raster size.
WORK_HEIGHT = 1400

#: Catalogue limit for the overlay rings: bright enough to recognise patterns,
#: few enough to keep the overlay readable.
OVERLAY_MAG_LIMIT = 4.5

#: The lens the rig actually carries (a Meike 6.5 mm circular fisheye on the
#: sample frames): the solver starts from this, while the overlay is *drawn*
#: with the ideal equiangular reference.  See
#: :func:`lens_calibration.selfcheck.nominal_lens`.
NOMINAL_PROJECTION = "equisolid"
NOMINAL_FOV_DEG = 190.0

#: Zenith distances of the alt-az grid rings drawn under the star circles.
GRID_THETA_DEG = (30.0, 60.0, 90.0)

#: Pinned stars needed before the *radial* model can be solved, i.e. before the
#: pins stop being a rigid on-screen fit and start being a lens calibration.
#: Four pins are eight equations for three unknowns (the optical centre and the
#: theta^3 term); each pin beyond that buys one more distortion coefficient, so
#: four stars buy ``k1``, five buy ``k1, k2`` and so on.  That is why the page
#: asks for a fourth star rather than a third: a rigid fit has three degrees of
#: freedom, so a second pin already exhausts it and a third can only report an
#: error it has no term to remove.
MIN_PINS_FOR_LENS = 4

#: Highest odd power the hand-pinned fit may use::
#:
#:     r = f * (theta + k1*theta^3 + k2*theta^5)
#:
#: ``cv2.fisheye`` itself carries ``k1 .. k4`` (up to theta^9), but those four
#: terms want dozens of well-spread calibration views: a handful of stars pinned
#: by hand on one frame cannot determine them, and the freedom they add is spent
#: folding the curve over just short of the rim.  Past this order more pins are
#: *redundancy* -- they add equations, and so tighten the fit, rather than
#: coefficients.
MAX_PIN_ORDER_DEG = 5

#: Distortion coefficients that order leaves free: ``k1`` (theta^3) and ``k2``
#: (theta^5).
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
    """The user's manual alignment of the raster to the sky overlay.

    The numbers are exactly the CSS transform the UI applies to the frame
    (``translate(dx, dy) rotate(roll) scaleX(+-1)`` about the frame centre):
    ``roll_deg`` turns the raster clockwise on screen, ``mirrored`` flips it
    left to right, and ``dx``/``dy`` shift it -- all in pixels of the work
    raster.

    The transform maps *raster* pixels to *overlay* pixels, so the overlay stays
    fixed (that is what "dragging the image under the overlay" means) and
    :meth:`inverse` is the map from the ideal sky overlay back onto the raster.
    """

    roll_deg: float = 0.0
    mirrored: bool = False
    dx: float = 0.0
    dy: float = 0.0

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

    ``star_x``/``star_y`` are overlay pixels -- the circle the user clicked
    first, i.e. where the ideal sky puts that star -- while ``x``/``y`` are
    raster pixels: the place the user found the same star in the photograph.
    The pair therefore belongs to both frames at once, which is what keeps the
    residual lines attached to the star while the transform is refitted, and it
    is the same convention as the solver's correspondences.
    """

    name: str
    star_x: float
    star_y: float
    x: float
    y: float

    def to_dict(self, residual_px: float = 0.0, draw: tuple[float, float] | None = None) -> dict:
        """JSON-ready pair; ``residual_px`` is what the refit still misses.

        ``draw`` is where the ring is drawn once a lens calibration exists --
        not the same place as ``star_x``/``star_y``, which stay the ideal sky
        position the pair was defined against (they carry the field angle).
        """
        payload = {
            "name": self.name,
            "star_x": float(self.star_x),
            "star_y": float(self.star_y),
            "x": float(self.x),
            "y": float(self.y),
            "residual_px": float(residual_px),
        }
        if draw is not None:
            payload["draw_x"] = float(draw[0])
            payload["draw_y"] = float(draw[1])
        return payload


@dataclass(frozen=True)
class PinnedLens:
    """Fisheye calibration solved from the stars the user pinned by hand.

    ``focal_px`` and ``coefficients`` are the model ``cv2.fisheye`` speaks::

        r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)   t = theta

    and ``lens`` is the same curve carried by the package's own
    ``theta(sigma)`` polynomial, so everything downstream (the solver, the
    overlay, the calibration artefact) keeps working unchanged.

    ``centre_px`` is in overlay pixels, i.e. in the frame the user pinned in;
    ``covers_sigma`` is the largest pinned radius as a fraction of the frame
    half height, which says whether the rim is measured or extrapolated.
    """

    focal_px: float
    coefficients: tuple[float, ...]
    centre_px: tuple[float, float]
    lens: LensModel
    rms_px: float
    max_px: float
    rms_deg: float
    covers_sigma: float
    notes: tuple[str, ...]
    points: tuple[dict, ...]
    predicted: tuple[tuple[float, float], ...]

    @property
    def num_coefficients(self) -> int:
        """Number of ``k`` terms the pins determined."""
        return len(self.coefficients)

    def camera(self, height_px: float) -> CameraModel:
        """Overlay-space camera that draws this calibration.

        No attitude: the pins already live in the overlay frame, so the fitted
        centre and lens are all that is needed to project the sky through them.
        """
        return CameraModel(
            lens=self.lens,
            xc=self.centre_px[0] / float(height_px),
            yc=self.centre_px[1] / float(height_px),
        )

    def to_dict(self) -> dict:
        """JSON-ready representation (the UI readout and the calibration artefact)."""
        return {
            "focal_px": float(self.focal_px),
            "coefficients": [float(k) for k in self.coefficients],
            "coefficients_unit": (
                "k1 .. kn in r = f*theta*(1 + k1 t^2 + k2 t^4 + ...), theta in radians"
            ),
            "num_coefficients": self.num_coefficients,
            "centre_px": {"x": float(self.centre_px[0]), "y": float(self.centre_px[1])},
            "rms_px": float(self.rms_px),
            "max_px": float(self.max_px),
            "rms_deg": float(self.rms_deg),
            "covers_sigma": float(self.covers_sigma),
            "pins": len(self.points),
            "notes": list(self.notes),
            "points": [dict(point) for point in self.points],
            "lens": self.lens.to_dict(),
        }


def pin_coefficient_count(
    pin_count: int, max_coefficients: int = MAX_PIN_COEFFICIENTS
) -> int:
    """Distortion terms a given number of pinned stars determines.

    Four pins buy the optical centre and ``k1`` (the theta^3 term); the fifth
    buys ``k2`` (theta^5), which is as far as the fit goes.  The rule is the one
    the count of unknowns gives: two centre coordinates plus ``n`` coefficients
    against two equations per pin, with the pins taken in the order they were
    placed -- so five stars is what makes the fit *redundant* rather than merely
    determined.  From there on the count is capped at
    :data:`MAX_PIN_COEFFICIENTS`, and every further pin is another equation for
    the same five unknowns.
    """
    return max(1, min(int(pin_count) - (MIN_PINS_FOR_LENS - 1), int(max_coefficients)))


def fit_alignment(
    points: Sequence[ControlPoint],
    width: float,
    height: float,
    roll_deg: float = 0.0,
    mirrored: bool = False,
) -> tuple[Alignment, np.ndarray]:
    """Rigid alignment that puts every pinned star on its circle.

    A similarity with unit scale -- rotation plus shift -- fitted by least
    squares through the paired points, which is the closed form of a
    georeferencing fit: with a single pin only the shift is determined (the
    incoming ``roll_deg`` is kept), with two the rotation follows exactly, and
    with more the fit is over-determined, so each pair keeps the residual that
    the ideal overlay and the real lens disagree by.  :meth:`AlignmentSession.solve`
    is what removes *that* -- this fit only moves the photograph.

    The mirror is not fitted.  A reflection cannot be recovered from a
    similarity fit without an arbitrary choice of hand, and the user has
    already said which hand the sky is in, so it is passed through.
    """
    points = list(points)
    if not points:
        raise ValueError("no stars pinned yet: click a circle, then the star in the photo")
    raster = np.array([[point.x, point.y] for point in points], dtype=np.float64)
    target = np.array([[point.star_x, point.star_y] for point in points], dtype=np.float64)

    # Fold the mirror in first and work about the frame centre, because that is
    # the order the CSS transform applies: mirror, rotate, then shift.
    centre = np.array([0.5 * float(width), 0.5 * float(height)])
    raster = raster - centre
    if mirrored:
        raster[:, 0] = -raster[:, 0]
    target = target - centre

    angle = math.radians(float(roll_deg))
    if len(points) > 1:
        # Kabsch in the plane: the rotation carrying the pins' spread onto the
        # circles' spread.  Coincident pins leave it to the incoming roll
        # instead of turning rounding noise into a rotation.
        a = raster - raster.mean(axis=0)
        b = target - target.mean(axis=0)
        cross = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
        dot = float(np.sum(a * b))
        if math.hypot(cross, dot) > 1e-12:
            angle = math.atan2(cross, dot)

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    shift = target.mean(axis=0) - rotation @ raster.mean(axis=0)
    alignment = Alignment(
        # Wrapped into the slider's range; R(theta + 360) is the same rotation,
        # so the shift fitted above stays valid.
        roll_deg=math.degrees((angle + math.pi) % (2.0 * math.pi) - math.pi),
        mirrored=bool(mirrored),
        dx=float(shift[0]),
        dy=float(shift[1]),
    )
    x, y = alignment.forward(
        np.array([point.x for point in points]),
        np.array([point.y for point in points]),
        width,
        height,
    )
    residual = np.hypot(
        x - np.array([point.star_x for point in points]),
        y - np.array([point.star_y for point in points]),
    )
    return alignment, residual


def fit_lens_from_pins(
    points: Sequence[ControlPoint],
    alignment: Alignment,
    width: float,
    height: float,
    ideal_lens: LensModel,
    *,
    coefficients: int | None = None,
    optimize_focal: bool = True,
) -> PinnedLens:
    """Solve the fisheye from the pinned stars: centre, focal length, ``k`` terms.

    The pins are the only data needed.  A circle in the overlay is drawn with the
    *ideal equiangular* lens, so its radius says what field angle the star was at
    (``theta = r / f_ideal``) and its direction says where; the click says where
    that star actually is once the photo is under the overlay (``alignment``
    puts it there).  Fitting::

        r = f * theta * (1 + k1 t^2 + k2 t^4)

    to those pairs is the model ``cv2.fisheye`` calibrates, and it is what a
    handful of stars can actually determine -- the coefficients enter linearly in
    the radius, so the pins past the fourth each buy another odd term, and the
    focal length carries the radial scale that :func:`fit_alignment` (a rigid
    similarity with unit scale) has no term for.  The series stops at
    theta**:data:`MAX_PIN_ORDER_DEG` (``k1`` and ``k2``): ``cv2.fisheye``'s
    ``k3`` and ``k4`` want a calibration rig's worth of views, and hand-pinned
    stars that tried to buy them would spend the freedom folding the curve over
    just short of the rim.  A sixth star and beyond therefore over-determine the
    same five unknowns, which is the point -- the fit averages their noise
    instead of chasing it.

    Why this is not ``match.fit_lens``: that fit needs a known attitude and a
    dozen matched detections, and it weights angles by 1/(dtheta/dsigma) to make
    the residuals mean pixels.  Here the residual *is* a pixel distance, so
    nothing has to be weighted, and the optical centre and focal length are free
    parameters instead of byproducts.

    The fit is bounded (``method="trf"``; Levenberg-Marquardt cannot take
    bounds): the centre stays within :data:`PIN_CENTRE_LIMIT` of the middle of
    the frame and each ``k`` within +-1.  Unbounded, four pins trade centre
    against ``k1`` until the polynomial folds over and the rim comes out tens of
    degrees wrong -- which is exactly the failure mode a minimum-pin fit invites.

    ``optimize_focal`` is on by default and the reason is measurable: the pins
    are overwhelmingly measuring the radial *scale*, so a frozen focal length
    (the overlay's own ``theta = r / f_ideal``) forces the first ``k`` to absorb
    the difference between the overlay's assumed field of view and the real one.
    On the sample frame the fitted focal length lands at 520 px/rad against the
    overlay's 440, and a four-pin fit with ``f`` frozen comes out at 21 px rms
    with a non-physical ``k1`` instead of 2 px with the right sign.  Freezing it
    is still useful when the focal length is known independently.
    """
    points = list(points)
    if len(points) < MIN_PINS_FOR_LENS:
        raise ValueError(
            f"the lens needs at least {MIN_PINS_FOR_LENS} pinned stars"
            f" (centre plus the theta^3 term), got {len(points)}"
        )
    count = (
        pin_coefficient_count(len(points))
        if coefficients is None
        else max(1, min(int(coefficients), MAX_PIN_COEFFICIENTS))
    )

    # The circles are the ideal sky: their radius in pixels divided by the
    # overlay's own pixels-per-radian is the field angle.
    focal_ideal = equiangular_pixels_per_radian(ideal_lens, height)
    ideal = np.array([[point.star_x, point.star_y] for point in points], dtype=np.float64)
    seen = np.column_stack(
        alignment.forward(
            np.array([point.x for point in points], dtype=np.float64),
            np.array([point.y for point in points], dtype=np.float64),
            width,
            height,
        )
    )
    offset = ideal - np.array([0.5 * float(width), 0.5 * float(height)])
    radius = np.hypot(offset[:, 0], offset[:, 1])
    if float(np.min(radius)) <= 0.0:
        raise ValueError(
            "a pinned star sits on the overlay centre, where no radius (and so no"
            " field angle) is defined: pin a star away from the middle"
        )
    direction = offset / radius[:, None]
    # Unchanging: a pin's field angle comes from the ideal projection, so it is
    # known before the fit and does not depend on the focal length being solved.
    theta = radius / focal_ideal

    def solve_once(terms: int):
        """Fit ``terms`` distortion coefficients; return (solution, error, drawn)."""

        def residuals(params: np.ndarray) -> np.ndarray:
            """Pixel error of every pin, in the order they were pinned.

            The field angle of a pin is fixed by the ideal projection -- that is
            what drawing the circles with the equiangular reference means -- and
            the free focal length only converts that angle into pixels.  Dividing
            by the *current* focal length instead would make the predicted radius
            ``r_ideal * (1 + k t^2)``, pinning the linear scale to the overlay's
            own radius and leaving ``f`` unable to represent a lens whose disc is
            not exactly as wide as the overlay assumes (2 % on the sample rig,
            more when the overlay field of view is a guess), so that no ``k``
            could fit it.
            """
            centre_x, centre_y = float(params[0]), float(params[1])
            index = 2
            focal = float(params[index]) if optimize_focal else focal_ideal
            if optimize_focal:
                index += 1
            t2 = theta * theta
            series = np.ones_like(theta)
            for order, k in enumerate(params[index : index + terms]):
                series = series + float(k) * t2 ** (order + 1)
            distorted = focal * theta * series
            return np.concatenate(
                [
                    centre_x + distorted * direction[:, 0] - seen[:, 0],
                    centre_y + distorted * direction[:, 1] - seen[:, 1],
                ]
            )

        limit = PIN_CENTRE_LIMIT * min(float(width), float(height))
        middle_x, middle_y = 0.5 * float(width), 0.5 * float(height)
        lower = [middle_x - limit, middle_y - limit]
        upper = [middle_x + limit, middle_y + limit]
        start = [middle_x, middle_y]
        if optimize_focal:
            lower.append(0.2 * focal_ideal)
            upper.append(5.0 * focal_ideal)
            start.append(focal_ideal)
        lower += [-1.0] * terms
        upper += [1.0] * terms
        start += [0.0] * terms
        return least_squares(
            residuals,
            np.array(start, dtype=np.float64),
            bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
            method="trf",
        )

    # A term the data cannot support does not fail gracefully: the extra freedom
    # is spent bending the curve until it folds over just short of the disc rim,
    # which is a lens that cannot be inverted where the sky is brightest.  So the
    # terms are dropped one at a time until the fitted curve covers the disc, and
    # the readout says which count was actually used.
    for terms in range(count, 0, -1):
        solved = solve_once(terms)
        index = 2
        focal = float(solved.x[index]) if optimize_focal else focal_ideal
        if optimize_focal:
            index += 1
        coefficients_solved = tuple(float(k) for k in solved.x[index : index + terms])
        centre = (float(solved.x[0]), float(solved.x[1]))
        try:
            lens = LensModel.from_fisheye(
                focal, coefficients_solved, height_px=height, sigma_edge=ideal_lens.sigma_edge
            )
            break
        except ValueError as error:
            if terms == 1:
                raise ValueError(
                    f"the pinned stars do not describe a physical lens ({error})"
                    " -- check the pins, or pin another star between them"
                ) from None
    else:  # pragma: no cover - the loop always breaks or raises
        raise ValueError("the pinned stars do not describe a physical lens")

    # Residuals, computed through the *stored* polynomial rather than the (f, k)
    # form the optimiser worked in: the page draws the ring where the converted
    # model puts it, and a line whose length disagreed with the reported number
    # would be a lie about the fit.  The conversion costs well under a pixel, and
    # charging it to the pins keeps the two honest.
    sigma_model = lens.sigma(theta)
    radius_model = sigma_model * 0.5 * float(height)
    predicted = np.column_stack(
        [
            centre[0] + radius_model * direction[:, 0],
            centre[1] + radius_model * direction[:, 1],
        ]
    )
    residual = np.hypot(predicted[:, 0] - seen[:, 0], predicted[:, 1] - seen[:, 1])
    sigma_seen = np.hypot(seen[:, 0] - centre[0], seen[:, 1] - centre[1]) / (0.5 * float(height))
    angle_error = np.degrees(lens.theta(sigma_seen) - theta)
    covers = float(np.max(radius) / (0.5 * float(height)))

    notes: list[str] = []
    if terms < count:
        notes.append(
            f"{len(points)} pins asked for {count} distortion terms, but {count} folds the"
            f" curve before the disc rim: fitted {terms} instead"
        )
    if terms == 1 and len(points) == MIN_PINS_FOR_LENS:
        notes.append(
            f"{len(points)} stars determine the centre and the theta^3 term only:"
            " pin a fifth to solve the theta^5 term as well"
        )
    elif terms == MAX_PIN_COEFFICIENTS and len(points) > MIN_PINS_FOR_LENS + terms - 1:
        notes.append(
            f"{len(points)} pins, but the fit stops at the theta^{MAX_PIN_ORDER_DEG}"
            " term: the stars past the fifth over-determine the same five unknowns"
            " rather than adding coefficients"
        )
    if covers < PIN_COVERAGE_WARN_SIGMA:
        notes.append(
            f"every pin is inside sigma {covers:.2f}: the rim is extrapolated,"
            " pin a star near the edge of the disc"
        )
    if not lens.monotonic():
        notes.append("the fitted curve is not monotone across the disc")

    return PinnedLens(
        focal_px=focal,
        coefficients=coefficients_solved,
        centre_px=centre,
        lens=lens,
        rms_px=float(np.sqrt(np.mean(np.square(residual)))),
        max_px=float(np.max(residual)),
        rms_deg=float(np.sqrt(np.mean(np.square(angle_error)))),
        covers_sigma=covers,
        notes=tuple(notes),
        points=tuple(
            point.to_dict(float(distance), (float(draw[0]), float(draw[1])))
            for point, distance, draw in zip(points, residual, predicted)
        ),
        predicted=tuple((float(x), float(y)) for x, y in predicted),
    )


@dataclass(frozen=True)
class SolveOutcome:
    """A solve plus the detections its correspondence indices refer to."""

    solution: AttitudeSolution
    detections: DetectedStars


class AlignmentSession:
    """Everything known about one frame, loaded lazily where it is expensive."""

    def __init__(
        self,
        path: str | Path,
        meta: ExposureMeta,
        gray: np.ndarray,
        frame_jpeg: bytes,
        circle: CircleFit,
        calibration_dir: str | Path = "calibrations",
    ) -> None:
        self.path = str(path)
        self.meta = meta
        self.gray = gray
        self.frame_jpeg = frame_jpeg
        self.circle = circle
        self.calibration_dir = Path(calibration_dir)
        self.utc_override: dt.datetime | None = None
        self.latitude_override: float | None = None
        self.longitude_override: float | None = None
        #: Last lens measured from the user's pins, if any.  It is a property of
        #: the rig rather than of the sky, so it survives a change of time/place
        #: and is only replaced (or dropped) by the next pin refit.
        self.pinned_lens: PinnedLens | None = None
        self._detections: DetectedStars | None = None
        self._field: StarField | None = None

    # --------------------------------------------------------------- loading
    @classmethod
    def open(
        cls,
        path: str | Path,
        work_height: int = WORK_HEIGHT,
        calibration_dir: str | Path = "calibrations",
    ) -> "AlignmentSession":
        """Read one frame: EXIF, a downscaled work raster, the image circle."""
        meta = read_exposure_meta(str(path))
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"cannot read {path}")
        scale = float(work_height) / float(frame.shape[0])
        size = (max(8, int(round(frame.shape[1] * scale))), int(round(work_height)))
        raster = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY)
        circle = fit_image_circle(gray, work_height=0)
        LOGGER.info(
            "%s: %dx%d work raster, disc centre %.4f/%.4f, sigma_edge %.3f",
            Path(path).name, raster.shape[1], raster.shape[0],
            circle.xc, circle.yc, circle.sigma_edge,
        )
        return cls(path, meta, gray, _preview_jpeg(raster), circle, calibration_dir)

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
    def detections(self) -> DetectedStars:
        """Point sources in the work raster (detected once)."""
        if self._detections is None:
            self._detections = detect_sources(self.gray)
            LOGGER.info("%s: %d detections", self.stem, len(self._detections))
        return self._detections

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
    def ideal_camera(
        self, projection: str = "equidistant", fov_deg: float = 180.0
    ) -> CameraModel:
        """Zenith-pointing camera for the *ideal* projection drawn as the overlay.

        ``equidistant`` (equiangular) with 180 deg across the disc is the
        reference the user aligns to: north up, east left, optical centre in the
        middle of the frame.  The disc is scaled to the measured image circle so
        the overlay rim lands on the real rim and the user can drag the disc
        onto it.
        """
        if projection not in FAMILIES:
            raise ValueError(f"unknown projection {projection!r}, expected one of {FAMILIES}")
        lens = LensModel.from_family(
            projection, fov_deg=float(fov_deg), sigma_edge=self.circle.sigma_edge
        )
        return CameraModel(lens=lens, xc=0.5, yc=0.5)

    def nominal_lens(self) -> LensModel:
        """The physically plausible starting lens for the solver.

        The overlay is drawn with the *ideal* equiangular reference, whose radii
        leave the real lens by several degrees at the rim.  Handing that model to
        the solver as its starting point throws away most of the matches before
        the lens can ever be refitted, so the solve starts from the nominal
        circular fisheye of the rig instead.
        """
        return LensModel.from_family(
            NOMINAL_PROJECTION, fov_deg=NOMINAL_FOV_DEG, sigma_edge=self.circle.sigma_edge
        )

    def seed_lens(self) -> LensModel:
        """Lens the solver starts from: the one the pins measured, if any.

        Four pinned stars already determine the centre and the first odd term, so
        once the user has aligned that many the solver never has to start from a
        guess: it starts from the measured projection and spends its iterations
        refining instead of searching.
        """
        return self.pinned_lens.lens if self.pinned_lens is not None else self.nominal_lens()

    def overlay(
        self,
        projection: str = "equidistant",
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
        camera: CameraModel | None = None,
    ) -> dict:
        """Bright catalogue stars as ringed positions in the overlay frame.

        ``camera`` re-projects the overlay through a *fitted* lens (see
        :func:`fit_lens_from_pins`).  The rings keep their ideal ``x``/``y`` --
        those are what a pin is defined against, and they are what carries the
        star's field angle -- while ``draw_x``/``draw_y`` say where the ring now
        belongs on screen.  The star *selection* stays on the ideal camera too,
        so a calibration can never make a pinned star disappear from the list.

        The ``aids`` layer (clusters, nebulae, the Milky Way) is drawn from the
        same lens but is never fitted; see :meth:`_aid_layer`.
        """
        ideal = self.ideal_camera(projection, fov_deg)
        fitted = ideal if camera is None else camera
        stars = self.field
        theta = np.arccos(np.clip(stars.enu @ ideal.rotation()[:, 2], -1.0, 1.0))
        keep = (stars.vmag <= float(mag_limit)) & (theta <= ideal.lens.theta_edge)
        alt = stars.alt[keep]
        az = stars.az[keep]
        x, y = ideal.pixel_from_altaz(alt, az, self.height)
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
                x, y, _take(stars.names, keep), stars.vmag[keep], alt, theta[keep]
            )
        ]
        if fitted is not ideal:
            draw_x, draw_y = fitted.pixel_from_altaz(alt, az, self.height)
            for ring, px, py in zip(rings, draw_x, draw_y):
                ring["draw_x"] = float(px)
                ring["draw_y"] = float(py)
        rings.sort(key=lambda ring: ring["vmag"])
        # Alt-az grid: rings at fixed zenith distance, plus the four cardinal
        # meridians.  Both come from the lens being drawn, so they stay exact
        # for whatever projection and field of view the user picked -- the JS
        # only has to scale them to CSS pixels.  A ring that would land on the
        # disc rim is dropped: the rim is already drawn dashed, and two lines in
        # the same place only make both harder to read.
        radius_edge = float(fitted.lens.sigma_edge * 0.5 * self.height)
        grid_rings = []
        for theta_deg in GRID_THETA_DEG:
            if theta_deg > fitted.lens.theta_edge_deg:
                continue
            radius = float(fitted.lens.sigma(math.radians(theta_deg)) * 0.5 * self.height)
            if radius > 0.995 * radius_edge:
                continue
            grid_rings.append({"theta_deg": float(theta_deg), "radius_px": radius})
        grid_meridians = []
        for az_deg, label in ((0.0, "N"), (90.0, "E"), (180.0, "S"), (270.0, "W")):
            px, py = fitted.pixel_from_altaz(
                np.array([0.0]), np.array([math.radians(az_deg)]), self.height
            )
            grid_meridians.append(
                {
                    "az_deg": float(az_deg),
                    "label": label,
                    # where the horizon sits, i.e. where the label goes
                    "x": float(px[0]),
                    "y": float(py[0]),
                }
            )
        centre_x, centre_y = fitted.centre_px(self.height)
        return {
            "projection": projection,
            "fov_deg": float(fov_deg),
            "calibrated": fitted is not ideal,
            "theta_edge_deg": float(fitted.lens.theta_edge_deg),
            "radius_px": float(fitted.lens.sigma_edge * 0.5 * self.height),
            "centre": {"x": float(centre_x), "y": float(centre_y)},
            "mag_limit": float(mag_limit),
            "count": len(rings),
            "grid": {"rings": grid_rings, "meridians": grid_meridians},
            "stars": rings,
            "aids": self._aid_layer(fitted),
        }

    def _aid_layer(self, camera: CameraModel) -> dict:
        """Clusters, nebulae and the Milky Way, projected for orientation only.

        These answer the question the rings cannot -- "is that smudge the
        Pleiades or the Double Cluster?" -- while the photo is still being
        dragged.  Nothing here can be fitted: a pin has to hang off a catalogue
        star, whose position is a point, whereas a cluster is a fuzzy patch with
        no centre worth clicking, so the layer is reported *beside* ``stars`` and
        the page's hit test never sees it.

        Like the grid, both the band and the markers are projected through the
        lens being drawn, so they follow a calibration instead of drifting off
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
        zenith = camera.rotation()[:, 2]
        theta_edge = float(camera.lens.theta_edge)

        theta = np.arccos(np.clip(enu_from_altaz(sky.alt, sky.az) @ zenith, -1.0, 1.0))
        keep = theta <= theta_edge
        x, y = camera.pixel_from_altaz(sky.alt[keep], sky.az[keep], self.height)
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
                x, y, theta[keep], sky.radius_deg[keep],
            )
        ]

        edge_alt = sky.band_alt.reshape(-1)
        edge_az = sky.band_az.reshape(-1)
        band_theta = np.arccos(
            np.clip(enu_from_altaz(edge_alt, edge_az) @ zenith, -1.0, 1.0)
        ).reshape(sky.band_alt.shape)
        inside = np.all(band_theta <= theta_edge, axis=0)
        band_x, band_y = camera.pixel_from_altaz(edge_alt, edge_az, self.height)
        band_x = band_x.reshape(sky.band_alt.shape)
        band_y = band_y.reshape(sky.band_alt.shape)
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

    # ------------------------------------------------- manual -> camera model
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
        projection: str = "equidistant",
        fov_deg: float = 180.0,
        lens: LensModel | None = None,
    ) -> CameraModel:
        """Camera model described by the user's manual alignment.

        The alignment says how the raster sits under the ideal overlay, so the
        stars' overlay positions (:meth:`Alignment.inverse`) are the raster
        positions they must have.  Finding the camera that predicts those
        positions is then a rigid-body fit, which
        :func:`~lens_calibration.match.update_attitude` solves in closed form --
        including the pixel convention, cross-checked by its improper-rotation
        test instead of by hand-derived signs.

        ``lens`` is the radial model the fit is expressed in.  It defaults to the
        overlay's ideal lens, which makes the result exactly reproduce the manual
        transform; pass :meth:`nominal_lens` for a seed the solver can use.
        """
        ideal = self.ideal_camera(projection, fov_deg)
        directions = self._overlay_directions(ideal)
        overlay_x, overlay_y = ideal.pixel_from_enu(directions, self.height)
        x, y = alignment.inverse(overlay_x, overlay_y, self.width, self.height)
        centre_x, centre_y = alignment.inverse(
            np.array([0.5 * self.width]), np.array([0.5 * self.height]),
            self.width, self.height,
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
        projection: str = "equidistant",
        fov_deg: float = 180.0,
    ) -> float:
        """Largest pixel error between ``camera`` and the manual alignment.

        Should be numerical noise (~1e-9 px); anything larger means the manual
        transform and the camera model disagree.
        """
        ideal = self.ideal_camera(projection, fov_deg)
        directions = self._overlay_directions(ideal)
        overlay_x, overlay_y = ideal.pixel_from_enu(directions, self.height)
        x, y = alignment.inverse(overlay_x, overlay_y, self.width, self.height)
        px, py = camera.pixel_from_enu(directions, self.height)
        return float(np.max(np.hypot(px - x, py - y)))

    # -------------------------------------------------------- live alignment
    def live_alignment(
        self,
        pins: Sequence[dict],
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
    ) -> dict:
        """Refit the manual transform -- and, from four pins, the lens -- through them.

        Each pin is a pair in the two frames the page already works in: the
        circle's place in the overlay and the pixel the user clicked in the
        photograph.  Every pair is refitted at once, so a new pin corrects the
        transform as a whole rather than nudging the photo.

        The first three pins only move the photo: :func:`fit_alignment` has three
        degrees of freedom, so a second pin already exhausts them and the pins
        from there on can only report an error they have no term to remove.
        From :data:`MIN_PINS_FOR_LENS` pins on, :func:`fit_lens_from_pins` takes
        over as well: the optical centre and the odd ``k`` terms are solved from
        the same pairs, each further pin buying one more term, and the response
        carries the re-projected overlay so the page can draw the circles through
        the lens that was just measured.
        """
        points: list[ControlPoint] = []
        for pin in pins:
            if not isinstance(pin, dict):
                raise ValueError("a control point must be an object")
            name = str(pin.get("name") or "").strip()
            if not name:
                raise ValueError("a control point needs the name of the star it pins")
            try:
                coordinates = [float(pin[key]) for key in ("star_x", "star_y", "x", "y")]
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"{name}: a control point needs star_x, star_y, x and y in pixels"
                ) from None
            points.append(ControlPoint(name, *coordinates))
        fitted, residual = fit_alignment(
            points, self.width, self.height, alignment.roll_deg, alignment.mirrored
        )
        rms = float(np.sqrt(np.mean(np.square(residual))))
        LOGGER.info(
            "%s: %d pinned star(s) refitted, %.2f px rms (worst %.2f px)",
            self.stem, len(points), rms, float(np.max(residual)),
        )
        response: dict = {
            "alignment": fitted.to_dict(),
            "points": [point.to_dict(error) for point, error in zip(points, residual)],
            "rms_px": rms,
            "max_px": float(np.max(residual)),
            "lens": None,
            "overlay": None,
        }
        if len(points) < MIN_PINS_FOR_LENS:
            self.pinned_lens = None
            return response

        # From here the pins can calibrate the lens as well.  A minimum-pin fit
        # can fail -- the centre and one odd term are only just determined -- and
        # that must not throw away the alignment the user has already won, so the
        # failure is reported beside the answer instead of in place of it.
        try:
            pinned = fit_lens_from_pins(
                points, fitted, self.width, self.height, self.ideal_camera(projection, fov_deg).lens
            )
        except ValueError as error:
            LOGGER.info("%s: pins do not calibrate the lens yet: %s", self.stem, error)
            self.pinned_lens = None
            response["lens_error"] = str(error)
            return response

        LOGGER.info(
            "%s: lens from %d pinned star(s): f %.1f px/rad, %d k term(s), %.2f px rms",
            self.stem, len(points), pinned.focal_px, pinned.num_coefficients, pinned.rms_px,
        )
        response["lens"] = pinned.to_dict()
        self.pinned_lens = pinned
        response["points"] = [dict(point) for point in pinned.points]
        response["rms_px"] = pinned.rms_px
        response["max_px"] = pinned.max_px
        response["overlay"] = self.overlay(
            projection, fov_deg, mag_limit, camera=pinned.camera(self.height)
        )
        return response

    # --------------------------------------------------------------- solving
    def solve(
        self,
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
        mag_limit: float = 5.5,
        **options,
    ) -> SolveOutcome:
        """Let the solver take over from the manual alignment.

        The aligned camera enters the candidate race as ``seed`` as well as
        being the fallback, so a good manual alignment can win on merit while
        the autonomous coarse ballot still gets its chance.  Detections are
        quality-filtered here rather than inside the solver, so the returned
        correspondence indices map straight back to pixels for the UI.
        """
        cam0 = self.camera_for(
            alignment, projection, fov_deg, lens=self.seed_lens()
        )
        if self.pinned_lens is not None:
            # The centre the pins measured is in overlay pixels; the seed camera
            # works in raster pixels, and the alignment is the map between them.
            pinned_x, pinned_y = alignment.inverse(
                np.array([self.pinned_lens.centre_px[0]]),
                np.array([self.pinned_lens.centre_px[1]]),
                self.width,
                self.height,
            )
            cam0 = cam0.with_updates(
                xc=float(pinned_x[0]) / self.height, yc=float(pinned_y[0]) / self.height
            )
        detections = self.detections.point_like()
        if not len(detections):
            detections = self.detections
        solution = solve_attitude(
            detections, self.field, cam0, self.height, self.width,
            mag_limit=mag_limit, quality_filter=False, seed=cam0, **options,
        )
        return SolveOutcome(solution=solution, detections=detections)

    def describe_solution(
        self,
        outcome: SolveOutcome,
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
    ) -> dict:
        """JSON payload for the UI: the solution plus the matched pairs."""
        solution = outcome.solution
        camera = solution.camera
        pairs = solution.correspondences
        detections = outcome.detections
        stars = self.field
        x = detections.x[pairs.det_index]
        y = detections.y[pairs.det_index]
        star_x, star_y = camera.pixel_from_altaz(
            stars.alt[pairs.star_index], stars.az[pairs.star_index], self.height
        )
        matched = [
            {
                "x": float(dx), "y": float(dy),
                "star_x": float(sx), "star_y": float(sy),
                "name": name, "vmag": float(mag), "sep_deg": float(sep),
            }
            for dx, dy, sx, sy, name, mag, sep in zip(
                x, y, star_x, star_y,
                _take(stars.names, pairs.star_index), stars.vmag[pairs.star_index],
                pairs.sep_deg,
            )
        ]
        matched.sort(key=lambda pair: pair["sep_deg"])
        seed = self.camera_for(alignment, projection, fov_deg, lens=self.nominal_lens())
        return {
            "solution": solution.to_dict(),
            "camera": _camera_dict(camera, self.height),
            "seed_camera": _camera_dict(seed, self.height),
            "transform_check_px": self.alignment_error_px(
                self.camera_for(alignment, projection, fov_deg), alignment, projection, fov_deg
            ),
            "pairs": matched,
            "detection_count": len(detections),
        }

    # ------------------------------------------------------------- artefacts
    def calibration_payload(
        self,
        outcome: SolveOutcome,
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
    ) -> dict:
        """Calibration artefact readable by :func:`~lens_calibration.model.read_calibration`."""
        camera = outcome.solution.camera
        return {
            "format": "lens_calibration/1",
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
            "camera": _camera_dict(camera),
            "overlay": {"projection": projection, "fov_deg": float(fov_deg)},
            "alignment": alignment.to_dict(),
            "solution": outcome.solution.to_dict(),
        }

    def calibration_path(self) -> Path:
        """Default artefact path for this frame."""
        return self.calibration_dir / f"{self.stem}.json"

    def save_calibration(
        self,
        outcome: SolveOutcome,
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
    ) -> Path:
        """Write the calibration artefact and return its path."""
        path = self.calibration_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.calibration_payload(outcome, alignment, projection, fov_deg)
        path.write_text(json.dumps(_plain(payload), indent=2) + "\n", encoding="utf-8")
        LOGGER.info("wrote %s", path)
        return path

    # ---------------------------------------------------------------- summary
    def summary(
        self,
        alignment: Alignment = Alignment(),
        projection: str = "equidistant",
        fov_deg: float = 180.0,
        mag_limit: float = OVERLAY_MAG_LIMIT,
    ) -> dict:
        """Everything the page needs on load."""
        overlay: dict | None = None
        error: str | None = None
        seed: dict | None = None
        transform_check: float | None = None
        try:
            overlay = self.overlay(projection, fov_deg, mag_limit)
        except ValueError as exc:
            error = str(exc)
        if overlay is not None:
            seed = _camera_dict(
                self.camera_for(alignment, projection, fov_deg, lens=self.nominal_lens()),
                self.height,
            )
            transform_check = self.alignment_error_px(
                self.camera_for(alignment, projection, fov_deg), alignment, projection, fov_deg
            )

        warnings: list[str] = []
        if "ASSUMED" in self.meta.time_source:
            warnings.append(
                "the EXIF clock carries no timezone offset, so it is being read as UTC;"
                " a camera set to local time is wrong by the UTC offset"
            )
        detections = self.detections
        return {
            "image": {
                "url": "/frame.jpg",
                "name": Path(self.path).name,
                "width": int(self.width),
                "height": int(self.height),
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
            "alignment": alignment.to_dict(),
            "seed_camera": seed,
            "transform_check_px": transform_check,
            "counts": {
                "detections": len(detections),
                "point_like": len(detections.point_like()),
                "catalogue": None if error else len(self.field),
            },
            "projections": list(FAMILIES),
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
    return {
        "xc": float(camera.xc),
        "yc": float(camera.yc),
        "centre_px": {"x": float(camera.xc * height), "y": float(camera.yc * height)},
        "roll_deg": float(math.degrees(camera.roll)),
        "tilt_x_deg": float(math.degrees(camera.tilt_x)),
        "tilt_y_deg": float(math.degrees(camera.tilt_y)),
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


def _preview_jpeg(raster: np.ndarray, gamma: float = 0.8, percentile: float = 99.5) -> bytes:
    """Brightness-stretched JPEG of the work raster for the page.

    A night frame is almost black; without a stretch the user cannot see the
    stars they are trying to line up.  The gain comes from a high percentile so
    one bright star does not set the scale of the whole frame.
    """
    data = raster.astype(np.float32)
    reference = float(np.percentile(data, percentile))
    if reference > 1.0:
        data *= min(255.0 / reference, 64.0)
    data = 255.0 * np.power(np.clip(data, 0.0, 255.0) / 255.0, gamma)
    ok, buffer = cv2.imencode(".jpg", data.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("could not encode the frame preview")
    return buffer.tobytes()
