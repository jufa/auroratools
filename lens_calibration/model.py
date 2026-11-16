"""Radial lens model: field angle as a function of normalised image radius.

Geometry conventions
--------------------
All image geometry is expressed in units of the **frame half height**, so a
calibration is independent of the raster size (full 4000 px frame, 2x2 binned
2000 px frame, downscaled preview):

    h       = H / 2                          frame half height in pixels
    u       = (x - xc_px) / h                image plane, +u = image right
    v       = (yc_px - y) / h                image plane, +v = image *up*
    sigma   = hypot(u, v)                    normalised radius; 1.0 is "the
                                             4000th pixel" of a 4000 px frame
    theta   field angle away from the optical axis (radians)

``xc``/``yc`` are the disc centre in pixels here;
:class:`~lens_calibration.geometry.CameraModel` stores the same centre as
fractions of the frame height and does the conversion.  Note that ``v`` points
up in the image while the camera frame's y axis points *down* the raster (see
:mod:`lens_calibration.geometry`).

The projection is modelled as an odd polynomial in sigma (odd *powers* only: a
rotationally symmetric lens cannot contain even powers of sigma, and any even
term would corrupt the slope on axis)::

    theta(sigma) = c1*sigma + c3*sigma**3 + c5*sigma**5 + c7*sigma**7

The model is linear in the coefficients, which is what makes the calibration a
closed-form least-squares problem once the camera attitude is known.

The image circle edge sits at ``sigma_edge`` (about 0.995 for the Meike 6.5 mm on
the ZV-E10).  The nominal 190 deg field of view enters as a soft prior
``theta(sigma_edge) == 95 deg``; see :mod:`lens_calibration.match`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import PchipInterpolator

#: Odd powers used by the model, in coefficient order.
POWERS: tuple[int, ...] = (1, 3, 5, 7)

#: Projection families usable as a physical first guess.  ``orthographic``
#: cannot represent more than 180 deg of field.
FAMILIES: tuple[str, ...] = ("equidistant", "equisolid", "orthographic", "stereographic")

#: Largest field angle the fisheye inverse search covers.  Past this the OpenCV
#: polynomial has folded over, so the radius no longer says anything about the
#: angle (and a fisheye that images beyond 179 deg does not exist).
FISHEYE_THETA_MAX_DEG = 179.0



def _as_float_array(value) -> np.ndarray:
    """Return ``value`` as a float64 array."""
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class LensModel:
    """Radial mapping ``theta(sigma)`` of a rotationally symmetric fisheye.

    Parameters
    ----------
    coeffs_deg:
        ``(c1, c3, c5, c7)`` in **degrees** per power of sigma, i.e.
        ``theta_deg(sigma) = sum_k c_k * sigma**(2k+1)``.  Degrees are used in
        the serialised form because they are readable; internal maths is in
        radians.
    sigma_edge:
        Normalised radius of the image circle edge (fraction of frame half
        height).  Anchors the 190 deg prior and the reported field angle.
    """

    coeffs_deg: tuple[float, float, float, float]
    sigma_edge: float = 0.995

    # ------------------------------------------------------------------ basics
    @property
    def coeffs_rad(self) -> tuple[float, ...]:
        """Coefficients in radians."""
        return tuple(math.radians(c) for c in self.coeffs_deg)

    @property
    def theta_edge(self) -> float:
        """Field angle at the image circle edge (radians)."""
        return float(self.theta(self.sigma_edge))

    @property
    def theta_edge_deg(self) -> float:
        """Field angle at the image circle edge (degrees)."""
        return math.degrees(self.theta_edge)

    def with_scale(self, scale: float) -> "LensModel":
        """Return the model with every field angle multiplied by ``scale``.

        This is the UI "zoom" control: scaling all coefficients scales the whole
        projection, which is exactly a radial zoom of the star pattern.
        """
        return LensModel(
            coeffs_deg=tuple(scale * c for c in self.coeffs_deg),  # type: ignore[arg-type]
            sigma_edge=self.sigma_edge,
        )

    # ------------------------------------------------------------- evaluation
    def theta_deg(self, sigma) -> np.ndarray:
        """Field angle in degrees for normalised radius ``sigma`` (array safe)."""
        s = _as_float_array(sigma)
        s2 = s * s
        c1, c3, c5, c7 = self.coeffs_deg
        # Horner in s^2; the leading factor s keeps the function odd.
        return s * (((c7 * s2 + c5) * s2 + c3) * s2 + c1)

    def theta(self, sigma) -> np.ndarray:
        """Field angle in radians for normalised radius ``sigma``."""
        return np.radians(self.theta_deg(sigma))

    def dtheta_dsigma(self, sigma) -> np.ndarray:
        """Radial magnification dtheta/dsigma (radians per unit sigma)."""
        s = _as_float_array(sigma)
        s2 = s * s
        c1, c3, c5, c7 = self.coeffs_rad
        return c1 + 3.0 * c3 * s2 + 5.0 * c5 * s2**2 + 7.0 * c7 * s2**3

    @cached_property
    def _inverse(self) -> PchipInterpolator:
        """Monotone cubic inverse of the forward model (cached per model)."""
        grid_sigma = self._sigma_grid()
        return PchipInterpolator(self.theta(grid_sigma), grid_sigma, extrapolate=True)

    def sigma(self, theta) -> np.ndarray:
        """Inverse mapping: normalised radius for a field angle in radians.

        A monotone cubic (PCHIP) interpolation of the forward model is used
        because the rim of the circle is heavily compressed and a polynomial
        inverse there is ill-conditioned.
        """
        return self._inverse(_as_float_array(theta))


    # ----------------------------------------------------------- sanity checks
    def _sigma_grid(self) -> np.ndarray:
        """Dense sigma grid covering a few percent beyond the image circle."""
        return np.linspace(0.0, max(1.05, 1.05 * self.sigma_edge), 2001)

    def monotonic(self, sigma_max: float | None = None) -> bool:
        """True if the mapping is strictly increasing up to ``sigma_max``."""
        limit = sigma_max if sigma_max is not None else max(1.05, 1.05 * self.sigma_edge)
        grid = np.linspace(0.0, limit, 512)
        return bool(np.all(self.dtheta_dsigma(grid) > 0.0))

    def monotonic_limit(self) -> float:
        """Largest sigma for which the model is still monotonically increasing."""
        grid = self._sigma_grid()
        bad = np.nonzero(self.dtheta_dsigma(grid) <= 0.0)[0]
        if bad.size == 0:
            return float(grid[-1])
        return float(grid[max(0, bad[0] - 1)])

    def describe(self) -> str:
        """Human readable one-line summary (logs, UI, calibration report)."""
        c1, c3, c5, c7 = self.coeffs_deg
        return (
            f"theta(sigma) = {c1:.4f} s {c3:+.4f} s^3 {c5:+.4f} s^5 {c7:+.4f} s^7 [deg]; "
            f"theta({self.sigma_edge:.3f}) = {self.theta_edge_deg:.2f} deg"
        )

    # -------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        """JSON-ready representation (degrees, with explicit units)."""
        return {
            "type": "odd_polynomial_theta_of_sigma",
            "sigma_normalisation": "sigma = r / (H/2), H = frame height in pixels",
            "powers": list(POWERS),
            "coeffs_unit": "deg",
            "coeffs": list(self.coeffs_deg),
            "theta_sigma_edge_deg": round(self.theta_edge_deg, 4),
            "sigma_edge": self.sigma_edge,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LensModel":
        """Rebuild a model from :meth:`to_dict` output."""
        coeffs = tuple(float(c) for c in payload["coeffs"])
        if len(coeffs) != len(POWERS):
            raise ValueError(f"expected {len(POWERS)} coefficients, got {len(coeffs)}")
        return cls(coeffs_deg=coeffs, sigma_edge=float(payload.get("sigma_edge", 0.995)))  # type: ignore[arg-type]

    # ------------------------------------------------------------ first guess
    @classmethod
    def from_family(
        cls,
        family: str = "equisolid",
        fov_deg: float = 190.0,
        sigma_edge: float = 0.995,
        samples: int = 400,
    ) -> "LensModel":
        """Fit the polynomial to a classic fisheye projection family.

        The family curve is normalised to pass exactly through
        ``(sigma_edge, fov_deg/2)``, then the odd polynomial is least-squares
        fitted to it.  This yields a physically plausible starting point --
        unlike extrapolating a polynomial that was fitted to the inner field.
        """
        theta_edge = math.radians(fov_deg / 2.0)
        scale = sigma_edge / _family_radius(family, np.array([theta_edge]))[0]

        theta = np.linspace(1e-6, theta_edge, samples)
        radius = _family_radius(family, theta) * scale

        grid = np.linspace(0.0, sigma_edge, samples)
        theta_of_sigma = np.interp(grid, radius, theta)
        design = np.column_stack([grid**p for p in POWERS])
        coeffs_rad, *_ = np.linalg.lstsq(design, theta_of_sigma, rcond=None)
        return cls(
            coeffs_deg=tuple(math.degrees(c) for c in coeffs_rad),  # type: ignore[arg-type]
            sigma_edge=sigma_edge,
        )

    @classmethod
    def from_fisheye(
        cls,
        focal_px: float,
        coefficients: tuple[float, ...] | list[float] = (),
        height_px: float = 2.0,
        sigma_edge: float = 0.995,
        samples: int = 400,
    ) -> "LensModel":
        """Fit the polynomial to an OpenCV fisheye curve ``(f, k1 .. k4)``.

        The hand-pinned fit solves the model ``cv2.fisheye`` uses -- focal length
        plus ``k1 .. k4`` in ``r = f*theta*(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4
        t^8)`` -- because that is the form the pins measure: the coefficients
        enter linearly in the radius, so stars past the fourth each buy another
        odd term -- up to ``k2``, where the pinned fit stops (theta^5, see
        :data:`lens_calibration.session.MAX_PIN_ORDER_DEG`) -- and the focal
        length carries the radial scale that a rigid on-screen fit has no term
        for.  The five coefficients stored here are this fixed basis fitted to
        the analytic curve, so they restate ``(f, k...)`` rather than adding
        freedom the pins never had.

        Everything else in this package speaks ``theta(sigma)``, so the solved
        curve is inverted onto the sigma grid and the odd polynomial fitted to
        it, exactly as :meth:`from_family` does for the analytic projections.
        Nothing is lost in the translation: both forms are odd series in theta,
        and a plausible ``k1 .. k4`` converts back to within 0.06 px over the
        disc (see :func:`fisheye_theta`).

        ``height_px`` sets the sigma normalisation, so pass the height of the
        raster ``focal_px`` was measured in.
        """
        half = 0.5 * float(height_px)
        if focal_px <= 0.0:
            raise ValueError(f"focal length must be positive, got {focal_px!r}")
        grid = np.linspace(0.0, float(sigma_edge), int(samples))
        theta_of_sigma = fisheye_theta(grid * half, focal_px, coefficients)
        design = np.column_stack([grid**p for p in POWERS])
        coeffs_rad, *_ = np.linalg.lstsq(design, theta_of_sigma, rcond=None)
        return cls(
            coeffs_deg=tuple(math.degrees(c) for c in coeffs_rad),  # type: ignore[arg-type]
            sigma_edge=float(sigma_edge),
        )


def fisheye_radius(theta, focal_px: float, coefficients=()) -> np.ndarray:
    """OpenCV fisheye image radius for a field angle in radians.

    ``r = f * theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)`` with ``t =
    theta``: the model ``cv2.fisheye`` calibrates and the one the hand-pinned
    fit solves.  ``coefficients = ()`` is the equidistant projection, which is
    why a plain equiangular lens needs no correction terms at all.
    """
    t = _as_float_array(theta)
    t2 = t * t
    series = np.ones_like(t)
    for index, coefficient in enumerate(coefficients):
        series = series + float(coefficient) * t2 ** (index + 1)
    return float(focal_px) * t * series


def fisheye_theta(radius_px, focal_px: float, coefficients=(), samples: int = 40001) -> np.ndarray:
    """Inverse of :func:`fisheye_radius`: field angle for an image radius.

    Inverted by interpolating the forward curve, which is monotone over the range
    a fisheye can actually image -- the same trick the polynomial model uses for
    its own inverse (:meth:`LensModel.sigma`).  Asking for a radius past the point
    where the polynomial folds over raises instead of silently clamping: that
    radius does not correspond to any angle, and a calibration that produces one
    is not physical.
    """
    theta = np.linspace(0.0, math.radians(FISHEYE_THETA_MAX_DEG), int(samples))
    radius = fisheye_radius(theta, focal_px, coefficients)
    rising = np.concatenate(([True], np.diff(radius) > 0.0))
    theta, radius = theta[rising], radius[rising]
    wanted = _as_float_array(radius_px)
    if np.any(wanted > radius[-1]):
        raise ValueError(
            f"radius {float(np.max(wanted)):.1f} px is past the fold of the fisheye"
            f" curve (max {radius[-1]:.1f} px): k1 .. k4 are not a physical lens"
        )
    return np.interp(wanted, radius, theta)


def _family_radius(family: str, theta: np.ndarray) -> np.ndarray:
    """Image radius (arbitrary focal length) for a field angle, per family."""
    if family == "equidistant":
        return theta
    if family == "equisolid":
        return 2.0 * np.sin(theta / 2.0)
    if family == "orthographic":
        return np.sin(theta)
    if family == "stereographic":
        return 2.0 * np.tan(theta / 2.0)
    raise ValueError(f"unknown projection family {family!r}, expected one of {FAMILIES}")


# ---------------------------------------------------------------- small utils
def read_calibration(path: str) -> dict:
    """Load a calibration JSON artefact."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def lens_from_calibration(payload: dict) -> LensModel:
    """Extract the lens model from a calibration artefact."""
    return LensModel.from_dict(payload["lens"])


def sigma_from_radius(radius_px: float, height_px: float) -> float:
    """Normalised radius for a pixel radius."""
    return float(radius_px / (height_px / 2.0))


def radius_from_sigma(sigma: float, height_px: float) -> float:
    """Pixel radius for a normalised radius."""
    return float(sigma * (height_px / 2.0))


def equiangular_pixels_per_radian(lens: LensModel, height_px: float) -> float:
    """Output scale keeping the undistorted disc the same size as the input.

    ``f_out = (sigma_edge * H/2) / theta_edge`` [pixels per radian], i.e. the
    image circle edge keeps its pixel radius while the field becomes
    equiangular.
    """
    theta_edge = lens.theta(lens.sigma_edge)
    return float(lens.sigma_edge * (height_px / 2.0) / theta_edge)

