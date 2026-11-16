"""Star detection, image-circle morphology and attitude solving.

Two independent problems are solved here:

1. **Geometry of the frame.**  A fisheye image is a disc: the lens model needs
   the disc centre and radius.  ``fit_image_circle`` finds them from the radial
   brightness step at the edge of the circle (and reports how elliptical the
   disc is -- a real ellipse means the frame was stretched during processing,
   which a radial polynomial cannot represent).

2. **Attitude.**  With the lens roughly known, the predicted positions of
   catalogue stars are matched against detected centroids: a coarse ballot gives
   the first guess, then nearest-neighbour iterations (ICP) with a closed-form
   Kabsch rotation update converge quickly because only the three attitude
   parameters *roll* and *tilt* are unknown -- the field angle of every pixel
   comes from the lens model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .catalog import StarField
from .geometry import CameraModel, attitude_from_rotation
from .model import POWERS, LensModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedStars:
    """Centroids of point sources, in pixels of the analysed raster."""

    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    fwhm: np.ndarray
    snr: np.ndarray
    height_px: float

    def __len__(self) -> int:
        return int(self.x.size)

    def brightest(self, count: int) -> "DetectedStars":
        """The ``count`` brightest detections (by flux)."""
        order = np.argsort(self.flux)[::-1][:count]
        return self.subset(order)

    def subset(self, index) -> "DetectedStars":
        """Detections selected by an index array or boolean mask."""
        return DetectedStars(
            x=self.x[index],
            y=self.y[index],
            flux=self.flux[index],
            peak=self.peak[index],
            fwhm=self.fwhm[index],
            snr=self.snr[index],
            height_px=self.height_px,
        )

    def point_like(
        self,
        min_snr: float = 6.0,
        fwhm_range: tuple[float, float] = (0.55, 2.2),
    ) -> "DetectedStars":
        """Detections that still look like stars (bright enough, PSF size sane).

        Real frames are full of things that are bright but not point sources --
        the Moon, planets, lamps, satellites, aurora filaments, tree edges.  A
        noise-limited detection sits just above the threshold, so requiring a
        comfortable signal-to-noise ratio and a PSF size within a band around
        the median of the surviving sample removes both ends without touching
        the star population.  Detection counts are therefore *reduced* on
        purpose; matching fractions are reported against that reference.
        """
        if not len(self):
            return self
        keep = self.snr >= min_snr
        if not np.any(keep):
            return self.subset(np.zeros(0, dtype=bool))
        typical = float(np.median(self.fwhm[keep]))
        if typical > 0.0:
            keep &= (self.fwhm >= fwhm_range[0] * typical) & (self.fwhm <= fwhm_range[1] * typical)
        return self.subset(keep)

    def scaled(self, height_px: float) -> "DetectedStars":
        """Rescale coordinates from ``self.height_px`` to another raster height."""
        factor = float(height_px) / float(self.height_px)
        if factor == 1.0:
            return self
        return DetectedStars(
            x=self.x * factor,
            y=self.y * factor,
            flux=self.flux,
            peak=self.peak,
            fwhm=self.fwhm * factor,
            snr=self.snr,
            height_px=float(height_px),
        )


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


def _local_background(gray: np.ndarray, block: int = 64) -> np.ndarray:
    """Smooth background estimate from a coarse block median.

    A night sky is a smooth gradient (twilight, light pollution, vignetting) and
    a median of 64 px blocks is insensitive to the stars inside them.
    """
    height, width = gray.shape[:2]
    small_w = max(4, width // block)
    small_h = max(4, height // block)
    coarse = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    coarse = ndimage.median_filter(coarse, size=5, mode="nearest")
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)


def detect_sources(
    gray: np.ndarray,
    threshold_sigma: float = 5.0,
    max_sources: int = 5000,
    min_fwhm: float = 1.2,
    max_area_factor: float = 4e-5,
    refine_radius: int = 8,
) -> DetectedStars:
    """Detect point sources in a 2-D raster.

    Works on the raw full-resolution frame or on a downscaled preview: all
    thresholds scale with the raster height, and the caller rescales the result
    with :meth:`DetectedStars.scaled`.

    Blended pairs are *not* deblended: at a 2.5 arcmin/pixel plate scale the
    catalogue stars are two orders of magnitude further apart than the PSF, so
    merged components are dominated by trees, satellites, planet discs and the
    Moon, which are rejected by shape and area instead.
    """
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("detect_sources expects a single channel raster")
    height, width = gray.shape
    residual = gray - _local_background(gray)
    noise = 1.4826 * float(np.median(np.abs(residual)))
    if not np.isfinite(noise) or noise <= 0.0:
        LOGGER.warning("flat frame, nothing to detect")
        return DetectedStars(*([np.zeros(0)] * 6), height_px=float(height))

    mask = (residual > threshold_sigma * noise).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return DetectedStars(*([np.zeros(0)] * 6), height_px=float(height))

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    max_area = max(24.0, max_area_factor * height * height)

    compact = areas <= min(200.0, max(max_area, 12.0))
    typical_area = float(np.median(areas[compact])) if np.any(compact) else 4.0
    min_area = max(2.0, 0.25 * typical_area)
    aspect = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1.0)
    fill = areas / np.maximum(widths * heights, 1.0)

    keep = (
        (areas >= min_area)
        & (areas <= max_area)
        & (aspect <= 3.0)
        & (fill >= 0.25)
    )
    if not np.any(keep):
        return DetectedStars(*([np.zeros(0)] * 6), height_px=float(height))

    component_ids = np.flatnonzero(keep) + 1
    flux = np.bincount(labels.ravel(), weights=residual.ravel(), minlength=count)[component_ids]

    # Flux-weighted sub-pixel centroids on background-subtracted patches.
    radius = int(refine_radius)
    xs = np.empty(component_ids.size)
    ys = np.empty(component_ids.size)
    peaks = np.empty(component_ids.size)
    fwhms = np.empty(component_ids.size)
    for index, component in enumerate(component_ids):
        left = stats[component, cv2.CC_STAT_LEFT]
        top = stats[component, cv2.CC_STAT_TOP]
        x0 = max(0, left - radius)
        y0 = max(0, top - radius)
        x1 = min(width, left + stats[component, cv2.CC_STAT_WIDTH] + radius)
        y1 = min(height, top + stats[component, cv2.CC_STAT_HEIGHT] + radius)
        patch = residual[y0:y1, x0:x1]
        weights = np.clip(patch, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            xs[index] = left
            ys[index] = top
            peaks[index] = 0.0
            fwhms[index] = 0.0
            continue
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        xs[index] = float((weights * grid_x).sum() / total)
        ys[index] = float((weights * grid_y).sum() / total)
        peaks[index] = float(weights.max())
        # Gaussian width from flux/peak (robust for an isolated star, and unlike
        # second moments it is not inflated by the noise in the patch wings).
        if peaks[index] > 0.0:
            fwhms[index] = 2.3548 * math.sqrt(max(total, 0.0) / (2.0 * math.pi * peaks[index]))
        else:
            fwhms[index] = 0.0

    snr = flux / (noise * np.sqrt(np.maximum(areas[keep], 1.0)))
    good = (fwhms >= min_fwhm) & np.isfinite(xs) & np.isfinite(ys)
    if not np.any(good):
        return DetectedStars(*([np.zeros(0)] * 6), height_px=float(height))

    order = np.argsort(flux[good])[::-1][:max_sources]
    return DetectedStars(
        x=xs[good][order],
        y=ys[good][order],
        flux=flux[good][order],
        peak=peaks[good][order],
        fwhm=fwhms[good][order],
        snr=snr[good][order],
        height_px=float(height),
    )


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


#: Pixels per radian of the plane used for nearest-neighbour matching.
MATCH_SCALE = 1000.0


@dataclass(frozen=True)
class Correspondence:
    """Matched (detection, catalogue star) pairs."""

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


def _zenith_plane(alt: np.ndarray, az: np.ndarray, scale: float = MATCH_SCALE):
    """Azimuthal-equidistant projection about the zenith, y pointing north-up."""
    radius = (0.5 * np.pi - np.asarray(alt)) * scale
    return radius * np.sin(az), -radius * np.cos(az)


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


def _rotation_about_vertical(angle: float) -> np.ndarray:
    """ENU rotation matrix that adds ``angle`` to every azimuth.

    Azimuth runs from north towards east, which is the opposite sense to a
    right-handed rotation about the up axis, hence the negated sine terms.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array(
        [[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _ballot_pointing(
    tilt_x: float,
    tilt_y: float,
    cam0: CameraModel,
    x: np.ndarray,
    y: np.ndarray,
    field: StarField,
    height_px: float,
    mag_limit: float,
    tolerance_deg: float,
) -> tuple[float, CameraModel]:
    """Score one pointing hypothesis and return it with the ballot roll applied."""
    cam = cam0.with_updates(tilt_x=tilt_x, tilt_y=tilt_y, roll=0.0)
    alt, az = cam.altaz_from_pixel(x, y, height_px)
    ballot = _roll_ballot(alt, az, field, mag_limit, tolerance_deg)
    if ballot is None:
        return 0.0, cam
    shift, score = ballot
    # The ballot measures a rotation about the *vertical*, which shifts every
    # azimuth equally for any pointing.  Rolling about the tilted optical axis
    # does not, so convert the rotation instead of assigning ``roll``.
    rotation = _rotation_about_vertical(shift) @ cam.rotation()
    roll, tilt_x_out, tilt_y_out = attitude_from_rotation(rotation)
    return score, cam.with_updates(roll=roll, tilt_x=tilt_x_out, tilt_y=tilt_y_out)


def _refine_pointing(
    tilt_x: float,
    tilt_y: float,
    evaluate: "Callable[[float, float], tuple[float, CameraModel]]",
    step_deg: float = 1.0,
    min_step_deg: float = 0.25,
    max_iterations: int = 8,
) -> tuple[float, CameraModel]:
    """Compass search on the ballot score; the objective is smooth nearby."""
    best_score, best_cam = evaluate(tilt_x, tilt_y)
    step = step_deg
    while step >= min_step_deg:
        for _ in range(max_iterations):
            improved = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                score, cam = evaluate(tilt_x + dx * step, tilt_y + dy * step)
                if score > best_score + 1e-6:
                    best_score, best_cam = score, cam
                    tilt_x += dx * step
                    tilt_y += dy * step
                    improved = True
                    break
            if not improved:
                break
        step *= 0.5
    return best_score, best_cam


def _nearest_pass(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    mag_limit: float,
    tolerance_deg: float,
    height_px: float,
    width_px: float,
) -> tuple[float, Correspondence]:
    """One projection plus nearest-neighbour pass, cost and matches together.

    The residual is the mean squared plane distance from every detection to its
    nearest catalogue star, clipped at the tolerance so that a detection with no
    counterpart (a planet, a satellite trail, a cloud edge) contributes a fixed
    amount instead of an unbounded one, and normalised to ``[0, 1]``.  Returning
    it alongside the matches lets a geometry search profile the attitude out
    without paying for a second matching pass.
    """
    empty = Correspondence(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    if not len(field) or not len(detections):
        return 1.0, empty

    # Indices stay global (into ``field``) so callers can look up ``field.enu``.
    selected = np.flatnonzero(field.vmag <= mag_limit)
    if selected.size == 0:
        return 1.0, empty
    alt = field.alt[selected]
    az = field.az[selected]

    pred_x, pred_y = cam.pixel_from_altaz(alt, az, height_px)
    inside = (pred_x > -0.02 * width_px) & (pred_x < 1.02 * width_px)
    inside &= (pred_y > -0.02 * height_px) & (pred_y < 1.02 * height_px)
    if not np.any(inside):
        return 1.0, empty

    subset = np.flatnonzero(inside)
    star_plane = np.column_stack(_zenith_plane(alt[subset], az[subset]))
    det_alt, det_az = cam.altaz_from_pixel(detections.x, detections.y, height_px)
    det_plane = np.column_stack(_zenith_plane(det_alt, det_az))

    limit = math.radians(tolerance_deg) * MATCH_SCALE
    star_distance, star_of_det = cKDTree(star_plane).query(det_plane, distance_upper_bound=limit)
    hit = np.isfinite(star_distance)
    squared = np.where(hit, star_distance * star_distance, limit * limit)
    cost = float(np.mean(squared) / (limit * limit))
    if not np.any(hit):
        return cost, empty

    _, det_of_star = cKDTree(det_plane).query(star_plane, distance_upper_bound=limit)
    candidates = np.flatnonzero(hit)
    nearest_star = star_of_det[candidates]
    mutual = det_of_star[nearest_star] == candidates
    det_index = candidates[mutual]
    star_index = selected[subset[nearest_star[mutual]]]
    return cost, Correspondence(
        det_index=det_index,
        star_index=star_index,
        sep_deg=np.degrees(star_distance[det_index] / MATCH_SCALE),
    )


def match_correspondences(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    mag_limit: float,
    tolerance_deg: float,
    height_px: float | None = None,
    width_px: float | None = None,
) -> Correspondence:
    """Mutually-nearest-neighbour matches between detections and predictions.

    Distances are measured in a zenith-centred azimuthal-equidistant plane, so
    the tolerance means the same thing everywhere in the frame (pixels do not:
    the plate scale varies by a factor of ~2 across a fisheye frame).
    """
    height = float(height_px or detections.height_px)
    width = float(width_px or height)
    _, correspondence = _nearest_pass(
        cam, detections, field, mag_limit, tolerance_deg, height, width
    )
    return correspondence


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


def _roll_ballot(
    alt_deg: np.ndarray,
    az_deg: np.ndarray,
    field: StarField,
    mag_limit: float,
    alt_tolerance_deg: float,
    n_bins: int = 360,
) -> tuple[float, float] | None:
    """Histogram the roll implied by every (detection, star) altitude match.

    For a fixed pointing each detection only needs an *azimuth offset* (the roll)
    to line up with a catalogue star; the offset that many pairs agree on is the
    answer.  The peak fraction doubles as a score for the pointing hypothesis.
    """
    stars = field.select(mag_limit)
    if not len(stars) or alt_deg.size == 0:
        return None
    order = np.argsort(stars.alt)
    star_alt = stars.alt[order]
    star_az = stars.az[order]

    tolerance = math.radians(alt_tolerance_deg)
    low = np.searchsorted(star_alt, alt_deg - tolerance)
    high = np.searchsorted(star_alt, alt_deg + tolerance)
    counts = high - low
    total = int(counts.sum())
    if total == 0:
        return None

    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    star_index = np.repeat(low, counts) + offsets
    det_index = np.repeat(np.arange(alt_deg.size), counts)
    shift = (star_az[star_index] - az_deg[det_index]) % (2.0 * np.pi)
    bin_index = np.minimum((shift / (2.0 * np.pi) * n_bins).astype(np.int32), n_bins - 1)
    histogram = np.bincount(bin_index, minlength=n_bins).astype(np.float64)

    # Circular 3-bin smoothing, then parabolic peak refinement.
    padded = np.concatenate([histogram[-1:], histogram, histogram[:1]])
    smooth = np.convolve(padded, np.ones(3) / 3.0, mode="valid")
    peak = int(np.argmax(smooth))
    left, centre, right = smooth[(peak - 1) % n_bins], smooth[peak], smooth[(peak + 1) % n_bins]
    denominator = left - 2.0 * centre + right
    sub_bin = 0.5 * (left - right) / denominator if abs(denominator) > 1e-9 else 0.0
    peak_shift = (peak + float(np.clip(sub_bin, -1.0, 1.0)) + 0.5) * (2.0 * np.pi / n_bins)
    score = float(centre / max(alt_deg.size, 1))
    return peak_shift % (2.0 * np.pi), score


def coarse_attitude_candidates(
    detections: DetectedStars,
    field: StarField,
    cam0: CameraModel,
    height_px: float | None = None,
    tilt_range_deg: float = 16.0,
    tilt_step_deg: float = 4.0,
    mag_limit: float = 4.5,
    alt_tolerance_deg: float | None = None,
    max_candidates: int = 3,
    seeds: int = 3,
    zoom_scan: bool = True,
    zoom_values: tuple[float, ...] = (0.92, 1.0, 1.08),
) -> list[tuple[float, CameraModel]]:
    """Global first guess: grid over the pointing, ballot over the roll.

    The frame must be roughly zenith pointing (the tolerance is the tilt range),
    which is exactly how a fisheye is used, and the sun is not needed for it.
    The grid is refined by a compass search on the ballot score, which is what
    brings the residual pointing error inside the ICP's basin of attraction.

    ``zoom_values`` rescales the radial polynomial during the same ballot.  The
    assumed field of view of a nominal lens is routinely 5-10 % out, which at
    190 deg is tens of degrees at the rim -- far more than any attitude can
    absorb -- so the bootstrap has to search it; the fitted lens in
    :func:`fit_lens` then replaces it.
    """
    height = float(height_px or detections.height_px)
    if alt_tolerance_deg is None:
        # The grid can miss the true axis by half a cell, so the ballot has to
        # tolerate that in altitude (it is far tighter in azimuth).
        alt_tolerance_deg = 0.75 * tilt_step_deg
    sample = detections.brightest(min(400, len(detections)))
    if not len(sample):
        return []

    def evaluate(tilt_x_deg: float, tilt_y_deg: float, zoom: float = 1.0):
        base = cam0 if zoom == 1.0 else cam0.with_updates(lens=cam0.lens.with_scale(zoom))
        return _ballot_pointing(
            math.radians(tilt_x_deg),
            math.radians(tilt_y_deg),
            base,
            sample.x,
            sample.y,
            field,
            height,
            mag_limit,
            alt_tolerance_deg,
        )

    def evaluator_for(zoom: float):
        def evaluate_at(tilt_x_deg: float, tilt_y_deg: float):
            return evaluate(tilt_x_deg, tilt_y_deg, zoom)

        return evaluate_at

    zooms = tuple(zoom_values) if zoom_scan else (1.0,)
    steps = np.arange(-tilt_range_deg, tilt_range_deg + 1e-9, tilt_step_deg)
    grid = [(evaluate(float(tx), float(ty), zoom)[0], float(tx), float(ty), zoom)
            for zoom in zooms for tx in steps for ty in steps]
    grid.sort(key=lambda item: item[0], reverse=True)

    candidates: list[tuple[float, CameraModel]] = []
    tried: list[tuple[float, float, float]] = []
    for score, tilt_x, tilt_y, zoom in grid[: seeds * 4]:
        if any(
            math.hypot(tilt_x - ox, tilt_y - oy) < tilt_step_deg and abs(zoom - oz) < 0.02
            for ox, oy, oz in tried
        ):
            continue
        tried.append((tilt_x, tilt_y, zoom))
        refined_score, cam = _refine_pointing(tilt_x, tilt_y, evaluator_for(zoom))
        candidates.append((max(refined_score, score), cam))
        if len(candidates) >= seeds:
            break

    candidates.sort(key=lambda item: item[0], reverse=True)
    distinct: list[tuple[float, CameraModel]] = []
    for score, cam in candidates:
        if all(
            abs(cam.tilt_x - other.tilt_x) > math.radians(1.5)
            or abs(cam.tilt_y - other.tilt_y) > math.radians(1.5)
            or abs(((cam.roll - other.roll + np.pi) % (2.0 * np.pi)) - np.pi) > math.radians(1.5)
            for _, other in distinct
        ):
            distinct.append((score, cam))
        if len(distinct) >= max_candidates:
            break
    return distinct


def _simplex(start: np.ndarray, steps: list[float]) -> np.ndarray:
    """Nelder-Mead initial simplex: ``start`` plus one axis step per parameter."""
    size = start.size
    extra = [start + np.eye(size)[index] * steps[index] for index in range(size)]
    return np.vstack([start, *extra])


def refine_geometry(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    height_px: float | None = None,
    width_px: float | None = None,
    mag_limit: float = 5.5,
    tolerance_deg: float = 1.0,
    free: tuple[str, ...] = ("xc", "yc", "scale"),
    max_iterations: int = 240,
) -> tuple[CameraModel, Correspondence, dict[str, float]]:
    """Fit the lens geometry (disc centre, radial scale) to the matched stars.

    The circle fit only sees the vignetting edge, which is a soft, often
    obstructed boundary, and a disc-centre error of a few pixels is a pointing
    error of a few arcminutes that no attitude can absorb.  The stars are a far
    sharper ruler: the attitude is profiled out analytically (Kabsch) and a
    Nelder-Mead search minimises the truncated nearest-neighbour residual over
    the requested geometry parameters.

    Parameters
    ----------
    free:
        Any of ``"xc"``, ``"yc"`` (disc centre, fractions of the frame height)
        and ``"scale"`` (multiplicative zoom of the lens polynomial).
    """
    height = float(height_px or detections.height_px)
    width = float(width_px or height)

    names = [name for name in ("xc", "yc", "scale") if name in free]
    if not names:
        correspondence = match_correspondences(
            cam, detections, field, mag_limit, tolerance_deg, height, width
        )
        return cam, correspondence, {}

    start = np.array([getattr(cam, name) if name in ("xc", "yc") else 1.0 for name in names])

    def unpack(values: np.ndarray) -> CameraModel:
        params = dict(zip(names, np.asarray(values, dtype=np.float64)))
        lens = cam.lens
        if "scale" in params:
            lens = lens.with_scale(float(params.pop("scale")))
        return cam.with_updates(lens=lens, **params)

    def objective(values: np.ndarray) -> float:
        candidate = unpack(values)
        if not (0.3 < candidate.xc < 0.7 and 0.3 < candidate.yc < 0.7):
            return 2.0
        if abs(candidate.lens.theta_edge_deg) > 130.0:
            return 2.0
        cost, correspondence = _nearest_pass(
            candidate, detections, field, mag_limit, tolerance_deg, height, width
        )
        # Profile the attitude out: the best rotation for this geometry is the
        # Kabsch solution of its own matches, so re-score the re-aimed camera.
        if len(correspondence) >= 4:
            candidate, _ = update_attitude(candidate, detections, field, correspondence, height)
            cost, _ = _nearest_pass(
                candidate, detections, field, mag_limit, tolerance_deg, height, width
            )
        return cost

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={
            "maxiter": max_iterations,
            "xatol": 2e-5,
            "fatol": 1e-5,
            "initial_simplex": _simplex(start, [3e-3, 3e-3, 1.5e-3]),
        },
    )
    fitted = unpack(result.x) if np.all(np.isfinite(result.x)) else cam
    if result.fun > objective(start):
        # Never return a worse geometry than the one supplied.
        fitted = cam
    fitted, _, _, _ = refine_attitude(
        fitted, detections, field, height, width, mag_limit=mag_limit,
        tolerance_deg=min(tolerance_deg, 1.0), iterations=12,
    )
    correspondence = match_correspondences(
        fitted, detections, field, mag_limit, 0.2, height, width
    )
    geometry = {
        "xc": fitted.xc,
        "yc": fitted.yc,
        "lens_scale": fitted.lens.coeffs_deg[0] / cam.lens.coeffs_deg[0],
        "theta_edge_deg": fitted.lens.theta_edge_deg,
        "cost": float(result.fun),
    }
    return fitted, correspondence, geometry


def _pair_field_angles(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    correspondences: Correspondence,
    height_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised radii of the matched pixels and the field angles they imply.

    The radius comes from the pixel position alone, so it is independent of the
    radial polynomial; the angle comes from the matched catalogue star through
    the current attitude.  That split is what makes the radial fit linear.
    """
    u, v = cam.plane_from_pixel(
        detections.x[correspondences.det_index],
        detections.y[correspondences.det_index],
        height_px,
    )
    d_cam = field.enu[correspondences.star_index] @ cam.rotation()
    return np.hypot(u, v), np.arccos(np.clip(d_cam[:, 2], -1.0, 1.0))


def fit_lens(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    height_px: float | None = None,
    width_px: float | None = None,
    mag_limit: float = 5.5,
    tolerance_deg: float = 0.5,
    iterations: int = 5,
    min_pairs: int = 8,
    prior_weight: float = 0.02,
) -> tuple[CameraModel, Correspondence, dict[str, object]]:
    """Fit the radial polynomial ``theta(sigma)`` of the lens to the matched stars.

    Every match is one equation ``sum_k c_k * sigma**(2k+1) = theta_obs`` with
    ``sigma`` known from the pixel and ``theta_obs`` known from the star, so one
    small linear least-squares solve replaces the nominal projection with a
    calibrated one.  Rows are weighted by the inverse slope
    ``1 / (dtheta/dsigma)`` so the fit minimises a *pixel* residual: the rim of
    the circle is compressed, so a given angle error there is worth more pixels
    than the same error on axis.  A ridge term pulls the coefficients back
    toward the current model: a handful of stars at the rim must not be able to
    fold the projection over.  Alternating the fit with the closed-form
    attitude update converges in a few rounds.
    """
    height = float(height_px or detections.height_px)
    width = float(width_px or height)
    sigma_edge = cam.lens.sigma_edge
    powers = np.array(POWERS, dtype=np.float64)

    best = cam
    correspondence = match_correspondences(
        best, detections, field, mag_limit, tolerance_deg, height, width
    )
    for _ in range(iterations):
        if len(correspondence) < min_pairs:
            break
        sigma, theta = _pair_field_angles(best, detections, field, correspondence, height)
        usable = (sigma > 0.05) & (sigma < 1.05 * sigma_edge)
        usable &= np.isfinite(theta) & (theta < math.radians(125.0))
        if np.count_nonzero(usable) < min_pairs:
            break
        sigma = sigma[usable]
        theta = theta[usable]

        design = sigma[:, None] ** powers[None, :]
        weight = 1.0 / np.maximum(best.lens.dtheta_dsigma(sigma), 1e-3)
        design = design * weight[:, None]
        column_scale = np.sqrt(np.mean(design * design, axis=0)) + 1e-12
        # Ridge strength relative to the data: ``prior_weight`` is the fraction
        # of a coefficient the prior is worth against the whole matched set.
        ridge = prior_weight * column_scale * math.sqrt(max(len(sigma), 1))
        previous = np.array(best.lens.coeffs_rad, dtype=np.float64)

        coefficients, *_ = np.linalg.lstsq(
            np.vstack([design, np.diag(ridge)]),
            np.concatenate([theta * weight, ridge * previous]),
            rcond=None,
        )
        if not np.all(coefficients > 0.0):
            break
        candidate = LensModel(
            coeffs_deg=tuple(math.degrees(float(c)) for c in coefficients),  # type: ignore[arg-type]
            sigma_edge=sigma_edge,
        )
        if not candidate.monotonic() or not (
            math.radians(30.0) < candidate.theta_edge < math.radians(125.0)
        ):
            break

        moved = float(np.max(np.abs(coefficients - previous)))
        best = best.with_updates(lens=candidate)
        correspondence = match_correspondences(
            best, detections, field, mag_limit, tolerance_deg, height, width
        )
        best, _ = update_attitude(best, detections, field, correspondence, height)
        correspondence = match_correspondences(
            best, detections, field, mag_limit, tolerance_deg, height, width
        )
        if moved < math.radians(0.01):
            break

    tight = match_correspondences(best, detections, field, mag_limit, 0.2, height, width)
    if len(tight):
        sigma, theta = _pair_field_angles(best, detections, field, tight, height)
        rms_deg = float(np.degrees(np.sqrt(np.mean(np.square(theta - best.lens.theta(sigma))))))
    else:
        rms_deg = float("nan")
    reference = cam.lens.coeffs_deg[0]
    lens = {
        "coeffs_deg": list(best.lens.coeffs_deg),
        "theta_edge_deg": best.lens.theta_edge_deg,
        "lens_scale": best.lens.coeffs_deg[0] / reference if reference else 1.0,
        "lens_rms_deg": rms_deg,
        "lens_pairs": float(len(tight)),
    }
    return best, tight, lens


@dataclass(frozen=True)
class AttitudeSolution:
    """Result of the star-based attitude solve for one frame."""

    camera: CameraModel
    correspondences: Correspondence
    mirrored: bool = False
    iterations: int = 0
    used_coarse: bool = True
    score: float = 0.0
    notes: tuple[str, ...] = ()
    geometry: dict[str, object] = field(default_factory=dict)

    @property
    def match_fraction(self) -> float:
        """Matched detections over all detections considered."""
        return self.score

    def to_dict(self) -> dict:
        """JSON-ready summary."""
        return {
            "matches": len(self.correspondences),
            "rms_deg": self.correspondences.rms_deg,
            "mirrored": self.mirrored,
            "iterations": self.iterations,
            "used_coarse": self.used_coarse,
            "score": self.score,
            "camera": {
                "xc": self.camera.xc,
                "yc": self.camera.yc,
                "roll_deg": math.degrees(self.camera.roll),
                "tilt_x_deg": math.degrees(self.camera.tilt_x),
                "tilt_y_deg": math.degrees(self.camera.tilt_y),
                "theta_edge_deg": self.camera.lens.theta_edge_deg,
            },
            "geometry": dict(self.geometry),
            "notes": list(self.notes),
        }


def refine_attitude(
    cam: CameraModel,
    detections: DetectedStars,
    field: StarField,
    height_px: float | None = None,
    width_px: float | None = None,
    mag_limit: float = 5.5,
    tolerance_deg: float = 3.0,
    iterations: int = 14,
    floor_deg: float = 0.2,
    shrink: float = 0.72,
) -> tuple[CameraModel, Correspondence, bool, int]:
    """Alternate matching and closed-form attitude updates until convergence.

    The first few rounds deliberately keep the widest tolerance: a coarse
    pointing guess is typically a couple of degrees out, and shrinking straight
    away would lock the matcher onto the wrong stars.
    """
    height = float(height_px or detections.height_px)
    correspondence = match_correspondences(
        cam, detections, field, mag_limit, tolerance_deg, height, width_px
    )
    for step in range(1, iterations + 1):
        if len(correspondence) < 4:
            break
        updated, _ = update_attitude(cam, detections, field, correspondence, height)
        moved = abs(updated.roll - cam.roll) + abs(updated.tilt_x - cam.tilt_x)
        moved += abs(updated.tilt_y - cam.tilt_y)
        cam = updated
        tolerance = max(floor_deg, tolerance_deg * shrink ** max(step - 3, 0))
        correspondence = match_correspondences(
            cam, detections, field, mag_limit, tolerance, height, width_px
        )
        if moved < math.radians(0.01) and tolerance <= floor_deg:
            return cam, correspondence, cam.mirrored, step
    return cam, correspondence, cam.mirrored, iterations


def solve_attitude(
    detections: DetectedStars,
    field: StarField,
    cam0: CameraModel,
    height_px: float | None = None,
    width_px: float | None = None,
    mag_limit: float = 5.5,
    coarse_mag_limit: float = 4.5,
    tilt_range_deg: float = 16.0,
    accept_rms_deg: float = 0.35,
    fit_geometry: bool = True,
    calibrate_lens: bool = True,
    quality_filter: bool = True,
    min_snr: float = 6.0,
    geometry_tolerance_deg: float = 1.0,
    coarse_mirror_score: float = 0.15,
    seed: CameraModel | None = None,
) -> AttitudeSolution:
    """Solve the camera attitude for one frame from the detected stars.

    Coarse-to-fine: a pointing grid with a roll ballot supplies the global
    first guesses (no sun or compass needed), the best few are refined by
    iterated nearest-neighbour matching plus Kabsch attitude updates, and the
    winner then has the lens geometry (disc centre, radial scale) fitted to the
    stars, which is finally followed by another attitude pass.  The radial
    polynomial itself is fitted last, from the matched stars, which is what
    turns the nominal projection into a calibration.

    ``quality_filter`` drops detections that cannot be stars (see
    :meth:`DetectedStars.point_like`) before anything else looks at them; the
    reported match fraction is relative to the surviving set.

    ``seed`` is an extra pointing hypothesis that competes with the ballot on
    merit -- the interactive UI puts the user's manual alignment here, so a good
    hand alignment does not have to beat the ballot to be used.

    ``coarse_mirror_score`` is the ballot score above which the second pixel
    convention (a mirrored raster) is skipped; the ballot score is reported in
    ``notes`` because a low one is the clearest sign that the frame's time,
    position or field of view -- not its attitude -- is wrong.
    """
    height = float(height_px or detections.height_px)
    notes: list[str] = []
    if not len(detections):
        return AttitudeSolution(cam0, Correspondence(np.zeros(0), np.zeros(0), np.zeros(0)),
                                notes=("no detections",))
    if quality_filter:
        reference = detections
        detections = detections.point_like(min_snr=min_snr)
        if not len(detections):
            notes.append("quality filter rejected every detection; using all of them")
            detections = reference
        elif len(detections) < len(reference):
            notes.append(f"quality filter kept {len(detections)} of {len(reference)} detections")

    # The coarse ballot assumes a normal pixel convention, so run it once per
    # convention and let the refinement decide which one explains the frame.
    candidates: list[tuple[float, CameraModel]] = []
    for index, mirrored in enumerate((False, True)):
        base = cam0.with_updates(mirrored=mirrored)
        found = coarse_attitude_candidates(
            detections, field, base, height,
            tilt_range_deg=tilt_range_deg, mag_limit=coarse_mag_limit,
        )
        candidates.extend(found)
        if index == 0 and found and found[0][0] >= coarse_mirror_score:
            # A healthy ballot score means the pixel convention is right: a
            # mirrored sky cannot be lined up by any single azimuth shift, so
            # the flipped pass is only worth its runtime when this one fails.
            break
    used_coarse = bool(candidates)
    if candidates:
        coarse_score = max(score for score, _ in candidates)
        notes.append(f"coarse ballot scored {coarse_score:.2f}")
    if not candidates:
        notes.append("coarse solve found nothing; refining the supplied attitude")
        candidates = [(0.0, cam0)]
    if seed is not None and all(candidate is not seed for _, candidate in candidates):
        # The caller's own pointing (the UI's manual alignment) is refined like
        # any ballot guess and then wins or loses on matches and RMS; the
        # tie-break score is arbitrary because it is only compared after both
        # have been refined.
        candidates.append((0.5, seed))

    total = float(len(detections))
    best: AttitudeSolution | None = None
    for guess_score, guess in candidates:
        cam, correspondence, mirrored, steps = refine_attitude(
            guess, detections, field, height, width_px,
            mag_limit=mag_limit, iterations=12,
        )
        score = len(correspondence) / total
        solution = AttitudeSolution(
            camera=cam,
            correspondences=correspondence,
            mirrored=mirrored,
            iterations=steps,
            used_coarse=used_coarse,
            score=score,
            notes=tuple(notes),
        )
        quality = (len(correspondence), -correspondence.rms_deg, guess_score)
        if best is None:
            best = solution
            best_quality = quality
        elif quality > best_quality:
            best, best_quality = solution, quality

    if best is None:
        return AttitudeSolution(cam0, Correspondence(np.zeros(0), np.zeros(0), np.zeros(0)),
                                notes=tuple(notes))
    geometry: dict[str, object] = {}
    if fit_geometry and len(best.correspondences) >= 20:
        cam, correspondence, fitted_geometry = refine_geometry(
            best.camera, detections, field, height, width_px,
            mag_limit=mag_limit, tolerance_deg=geometry_tolerance_deg,
        )
        geometry.update(fitted_geometry)
        total_matches = len(correspondence)
        best = AttitudeSolution(
            camera=cam, correspondences=correspondence, mirrored=cam.mirrored,
            iterations=best.iterations, used_coarse=best.used_coarse,
            score=total_matches / total, notes=tuple(notes), geometry=geometry,
        )
    if calibrate_lens and len(best.correspondences) >= 12:
        cam, correspondence, lens = fit_lens(
            best.camera, detections, field, height, width_px,
            mag_limit=mag_limit, tolerance_deg=max(geometry_tolerance_deg, 0.5),
        )
        cam, correspondence, mirrored, steps = refine_attitude(
            cam, detections, field, height, width_px, mag_limit=mag_limit, iterations=10,
        )
        geometry.update(lens)
        best = AttitudeSolution(
            camera=cam, correspondences=correspondence, mirrored=mirrored,
            iterations=best.iterations + steps, used_coarse=best.used_coarse,
            score=len(correspondence) / total, notes=tuple(notes), geometry=geometry,
        )
        notes.append(
            f"lens fitted on {len(correspondence)} pairs, "
            f"{float(lens['lens_rms_deg']):.3f} deg radial RMS"
        )
    elif calibrate_lens:
        notes.append("too few matches to fit the lens; using the nominal projection")
    if best.correspondences.rms_deg > accept_rms_deg:
        notes.append(f"RMS {best.correspondences.rms_deg:.3f} deg exceeds acceptance")
    if len(best.correspondences) < 20:
        notes.append("few matches; solve may be unreliable")
    if notes:
        best = AttitudeSolution(
            camera=best.camera, correspondences=best.correspondences,
            mirrored=best.mirrored, iterations=best.iterations,
            used_coarse=best.used_coarse, score=best.score, notes=tuple(notes),
            geometry=best.geometry,
        )
    return best


