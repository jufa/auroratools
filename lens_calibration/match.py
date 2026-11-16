"""Image-circle morphology and the closed-form attitude update.

Two independent problems live here:

1. **Geometry of the frame.**  A fisheye image is a disc: the lens model needs
   the disc centre and radius.  ``fit_image_circle`` finds them from the radial
   brightness step at the edge of the circle (and reports how elliptical the
   disc is -- a real ellipse means the frame was stretched during processing,
   which a radial polynomial cannot represent).

2. **Attitude.**  Given a handful of (pixel, sky direction) pairs, the rotation
   carrying one onto the other is closed form (Kabsch), and
   :func:`update_attitude` turns it into the three attitude parameters *roll*
   and *tilt* -- the field angle of every pixel already comes from the lens
   model, so nothing about the pointing is searched for.  The session's manual
   alignment is converted into a camera this way, and so is every pin refit.

There is no star detection and no matching here any more: the stars are chosen
by hand on the page, which knows which star is which and needs no ballot to say
so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .catalog import StarField
from .geometry import CameraModel, attitude_from_rotation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedStars:
    """Pixel positions paired with sky directions (a fit's input, not a catalogue).

    ``flux``/``peak``/``fwhm``/``snr`` are carried for the sake of callers that
    have them; nothing in this package detects sources any more.
    """

    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    fwhm: np.ndarray
    snr: np.ndarray
    height_px: float

    def __len__(self) -> int:
        return int(self.x.size)


@dataclass(frozen=True)
class CircleFit:
    """Image circle geometry as fractions of the frame height."""

    xc: float
    yc: float
    sigma_edge: float
    sector_sigma: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    sector_angle: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    contrast: float = 0.0
    confidence: float = 0.0

    @property
    def diameter_fraction(self) -> float:
        """Disc diameter as a fraction of the frame height."""
        return 2.0 * self.sigma_edge

    @property
    def axis_ratio(self) -> float:
        """Ratio of the shortest to longest radius (1.0 = circular disc)."""
        if self.sector_sigma.size < 6:
            return 1.0
        usable = self.sector_sigma[np.isfinite(self.sector_sigma) & (self.sector_sigma > 0)]
        if usable.size < 6:
            return 1.0
        return float(np.percentile(usable, 10) / np.percentile(usable, 90))

    def to_dict(self) -> dict:
        """JSON-ready summary for the UI."""
        return {
            "xc_frac": self.xc,
            "yc_frac": self.yc,
            "sigma_edge": self.sigma_edge,
            "diameter_fraction": self.diameter_fraction,
            "axis_ratio": self.axis_ratio,
            "contrast": self.contrast,
            "confidence": self.confidence,
        }



def _fit_circle_kasa(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit through ``(N, 2)`` boundary points."""
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    cx, cy = float(solution[0]), float(solution[1])
    radius_sq = solution[2] + cx * cx + cy * cy
    return cx, cy, float(math.sqrt(max(radius_sq, 0.0)))


def _profiles(gray: np.ndarray, cx: float, cy: float, n_sectors: int, n_bins: int):
    """Radial profiles about ``(cx, cy)``: overall and per angular sector."""
    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    sector = ((angle + np.pi) / (2.0 * np.pi) * n_sectors).astype(np.int32) % n_sectors
    bin_index = np.minimum((radius / (0.5 * height) * n_bins).astype(np.int32), n_bins - 1)
    combo = (sector * n_bins + bin_index).ravel()
    sums = np.bincount(combo, weights=gray.ravel(), minlength=n_sectors * n_bins)
    counts = np.bincount(combo, minlength=n_sectors * n_bins)
    return (sums / np.maximum(counts, 1)).reshape(n_sectors, n_bins), radius


def fit_image_circle(
    gray: np.ndarray,
    work_height: int = 1000,
    n_sectors: int = 30,
    n_bins: int = 200,
    min_contrast: float = 0.08,
) -> CircleFit:
    """Locate the image circle (disc centre, radius, ellipticity).

    The lens vignettes to black outside the image circle, so the disc boundary is
    a strong radial brightness step.  It is found in every angular sector, which
    both hardens the fit against a tree or a rooftop blocking part of the rim and
    exposes the axis ratio: a fisheye disc can only be *elliptical* if the frame
    was resampled with a different aspect ratio, which no radial polynomial can
    absorb -- hence the diagnostic.
    """
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("fit_image_circle expects a single channel raster")
    height, width = gray.shape
    scale = 1.0
    if work_height and height > work_height:
        scale = work_height / float(height)
        gray = cv2.resize(
            gray,
            (max(8, int(round(width * scale))), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = gray.shape

    cx, cy = width / 2.0, height / 2.0
    corner = float(np.median(gray[: height // 12, : width // 12]))
    centre_level = float(np.median(gray[height // 2 - 20 : height // 2 + 20, width // 2 - 20 : width // 2 + 20]))
    contrast = (centre_level - corner) / max(abs(centre_level), 1e-6)
    if contrast < min_contrast:
        LOGGER.warning("no dark vignetted corners: disc fills the frame (contrast %.3f)", contrast)
        return CircleFit(xc=0.5, yc=0.5, sigma_edge=1.0, contrast=contrast, confidence=0.0)

    mid = 0.5 * (centre_level + corner)
    sector_angles = np.zeros(0)
    sector_sigmas = np.zeros(0)
    for _ in range(4):
        profiles, _ = _profiles(gray, cx, cy, n_sectors, n_bins)
        bin_width = 0.5 * height / n_bins  # pixels per radial bin
        above = profiles > mid
        last_index = n_bins - 1 - np.argmax(above[:, ::-1], axis=1)
        usable = np.flatnonzero(above.any(axis=1) & (last_index < n_bins - 2))
        if usable.size < 6:
            LOGGER.warning("could not locate the image circle (%d usable sectors)", usable.size)
            return CircleFit(xc=0.5, yc=0.5, sigma_edge=1.0, contrast=contrast, confidence=0.0)
        last = last_index[usable]
        rows = np.arange(usable.size)
        before = profiles[usable, last]
        after = profiles[usable, np.minimum(last + 1, n_bins - 1)]
        fraction = np.clip(
            (before - mid) / np.maximum(before - after, 1e-6), 0.0, 1.0
        )
        radii = (last + fraction) * bin_width
        theta = (usable + 0.5) / n_sectors * 2.0 * np.pi - np.pi
        points = np.column_stack(
            [cx + radii * np.cos(theta), cy + radii * np.sin(theta)]
        )
        new_cx, new_cy, radius = _fit_circle_kasa(points)
        moved = math.hypot(new_cx - cx, new_cy - cy)
        cx, cy = new_cx, new_cy
        sector_angles = theta
        sector_sigmas = radii / (0.5 * height)
        if moved < 0.05:
            break

    half = 0.5 * height
    fit = CircleFit(
        xc=float(cx) / height,
        yc=float(cy) / height,
        sigma_edge=float(radius / half),
        sector_sigma=sector_sigmas,
        sector_angle=sector_angles,
        contrast=float(contrast),
        confidence=float(min(1.0, sector_angles.size / n_sectors) * min(1.0, contrast / 0.4)),
    )
    return fit


@dataclass(frozen=True)
class Correspondence:
    """Index pairs: ``det_index[i]`` is the pixel paired with ``star_index[i]``."""

    det_index: np.ndarray
    star_index: np.ndarray
    sep_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.det_index.size)

    @property
    def rms_deg(self) -> float:
        """RMS separation of the matched pairs (degrees)."""
        if not len(self):
            return float("inf")
        return float(np.sqrt(np.mean(self.sep_deg ** 2)))

    def to_dict(self) -> dict:
        """JSON-ready summary."""
        return {
            "n": len(self),
            "rms_deg": self.rms_deg,
            "median_sep_deg": float(np.median(self.sep_deg)) if len(self) else None,
            "max_sep_deg": float(np.max(self.sep_deg)) if len(self) else None,
        }

def _cam_vectors(cam: CameraModel, x: np.ndarray, y: np.ndarray, height_px: float) -> np.ndarray:
    """Camera-frame unit vectors of the given pixels (attitude independent).

    Delegates to :meth:`CameraModel.camera_vectors` so the camera frame's
    handedness (x right, y *down* the raster, z along the optical axis) is
    defined in exactly one place.
    """
    return cam.camera_vectors(x, y, height_px)


def _kabsch(points_cam: np.ndarray, vectors_enu: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rotation taking camera-frame unit vectors onto ENU unit vectors.

    Returns the rotation and whether the raw optimum was *improper* (a
    reflection), which is how an image-directory flip shows up.
    """
    covariance = points_cam.T @ vectors_enu
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    mirrored = bool(np.linalg.det(rotation) < 0.0)
    if mirrored:
        vt_matrix = vt_matrix.copy()
        vt_matrix[2, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    return rotation, mirrored


def rotation_from_pairs(
    camera_vectors: np.ndarray, sky_vectors: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Closed-form rotation taking camera-frame vectors onto sky directions.

    :func:`update_attitude` is the matched-pairs form of this: it has to find
    which star goes with which detection first.  A pin says which star it is, so
    the hand-pinned fit knows both halves of every pair and can take the answer
    straight from Kabsch.

    Returns the rotation and whether the raw optimum was *improper* -- a
    reflection, which is how a raster that is a mirror image of the sky shows up
    (an image-directory flip, a mirrored debayer).  Two pairs are enough to
    determine a rotation; three are enough to trust the handedness test, which is
    why the caller only acts on it from :data:`~lens_calibration.session.
    MIN_PINS_FOR_MIRROR` pins on.
    """
    return _kabsch(
        np.asarray(camera_vectors, dtype=np.float64),
        np.asarray(sky_vectors, dtype=np.float64),
    )


def update_attitude(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    correspondences: Correspondence,
    height_px: float | None = None,
) -> tuple[CameraModel, bool]:
    """Closed-form attitude update (Kabsch) from the matched pairs.

    Returns the updated camera model and whether the frame turned out mirrored.
    """
    if len(correspondences) < 3:
        return cam, cam.mirrored
    height = float(height_px or detections.height_px)
    x = detections.x[correspondences.det_index]
    y = detections.y[correspondences.det_index]
    vectors = field.enu[correspondences.star_index]

    rotation, improper = _kabsch(_cam_vectors(cam, x, y, height), vectors)
    if improper:
        # An image-directory or mirror flip shows up as an improper rotation:
        # the pixel convention is the wrong way round, so toggle it and refit.
        cam = cam.with_updates(mirrored=not cam.mirrored)
        rotation, _ = _kabsch(_cam_vectors(cam, x, y, height), vectors)
    roll, tilt_x, tilt_y = attitude_from_rotation(rotation)
    return cam.with_updates(roll=roll, tilt_x=tilt_x, tilt_y=tilt_y), cam.mirrored
