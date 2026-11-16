"""Camera geometry: attitude, pixel <-> sky conversions and equiangular output.

Frames
------
``ENU``    world frame, x = east, y = north, z = up (right handed).
``camera`` camera frame, x = image right, y = image **down**, z = optical axis
           (pointing into the scene) -- the standard pinhole convention, which
           is right handed and therefore what a real (unmirrored) photograph
           looks like.  Together with a rotation matrix whose columns are those
           three axes expressed in ENU, this frame stays right handed, so a
           raster that was mirrored during processing shows up as
           ``det(R) = -1`` instead of being silently absorbed into the attitude.

           Using y = image *up* in a right-handed frame would be a left-handed,
           i.e. mirrored, description of the same camera: it consistently puts
           east and west the wrong way round, so the star pattern is reflected
           and no attitude can ever match it.

Attitude
--------
The optical axis is tilted away from the zenith by a small tangent-plane offset
``(tilt_x, tilt_y)`` (radians toward east and north respectively) and the
camera is then rotated about its own axis by ``roll``.  ``roll = 0`` puts
celestial north at the top of the frame; a positive roll rotates the *sky* on
the frame clockwise.

Everything pixel related takes the frame height ``H`` and works in the
dimensionless units of :mod:`lens_calibration.model` (fractions of ``H/2``), so
the same camera model applies to full, binned and preview rasters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .model import LensModel, equiangular_pixels_per_radian

LOGGER = logging.getLogger(__name__)

#: Directions of the ENU basis in world coordinates.
ENU_EAST = np.array([1.0, 0.0, 0.0])
ENU_NORTH = np.array([0.0, 1.0, 0.0])
ENU_UP = np.array([0.0, 0.0, 1.0])


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` normalised along the last axis."""
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, 1e-12)


@lru_cache(maxsize=64)
def radial_lut(lens: LensModel, height_px: float, rho_max: float, samples: int = 4096):
    """Cached 1-D table: equiangular output radius fraction -> input radius (px).

    The optical-axis equiangular resampling is radially symmetric, so the whole
    lens evaluation collapses into this single table; the 2-D map is then built
    with a linear interpolation over the radius only.
    """
    rho = np.linspace(0.0, max(rho_max, 1.05 * lens.sigma_edge), samples)
    theta = rho * (lens.theta_edge / lens.sigma_edge)
    return rho, lens.sigma(theta) * (height_px / 2.0)


@lru_cache(maxsize=32)
def ideal_radius_lut(camera, ideal, height_px: float, samples: int = 4096):
    """Cached 1-D table: ideal output radius (px) -> photograph radius (px).

    The inverse of :func:`radial_lut`, and for the same reason: mapping one lens
    onto another is radially symmetric, so the whole composition -- ideal angle
    for an output radius, then the radius ``camera`` puts that angle at -- is one
    monotone table over the radius from the optical centre.  Both curves are
    anchored at the same rim, so the table ends where the image circle does.
    """
    half = height_px / 2.0
    radius = np.linspace(0.0, ideal.lens.sigma_edge * half, samples)
    theta = ideal.lens.theta(radius / half)
    return radius, camera.lens.sigma(theta) * half



def enu_from_altaz(alt_rad, az_rad) -> np.ndarray:
    """Unit vector(s) in ENU for altitude/azimuth in radians (az from north)."""
    alt = np.asarray(alt_rad, dtype=np.float64)
    az = np.asarray(az_rad, dtype=np.float64)
    cos_alt = np.cos(alt)
    return np.stack(
        [cos_alt * np.sin(az), cos_alt * np.cos(az), np.sin(alt)], axis=-1
    )


def altaz_from_enu(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Altitude and azimuth (radians, az from north, east positive) of a vector."""
    d = _unit(np.asarray(direction, dtype=np.float64))
    alt = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    az = np.mod(np.arctan2(d[..., 0], d[..., 1]), 2.0 * np.pi)
    return alt, az


def rotation_from_attitude(roll: float, tilt_x: float, tilt_y: float) -> np.ndarray:
    """Rotation matrix taking camera coordinates to ENU.

    Parameters
    ----------
    roll:
        Rotation about the optical axis (radians).  It is applied *after* the
        tilt, so ``0`` means "celestial north up in the frame" for the tilted
        axis as well.
    tilt_x, tilt_y:
        Tangent-plane offsets of the optical axis toward east / north (radians).
    """
    optical_axis = _unit(np.array([tilt_x, tilt_y, 1.0]))
    up0 = ENU_NORTH - np.dot(ENU_NORTH, optical_axis) * optical_axis
    if np.linalg.norm(up0) < 1e-6:  # camera pointing at the celestial pole
        up0 = ENU_EAST - np.dot(ENU_EAST, optical_axis) * optical_axis
    up0 = _unit(up0)

    # Rotate the "north up" reference about the optical axis by ``roll``.
    up = up0 * math.cos(roll) + np.cross(optical_axis, up0) * math.sin(roll)
    up = _unit(up)

    # ``up`` is the direction that appears at the top of the frame; the camera
    # frame's y axis points *down* the raster (pinhole convention).
    y_axis = -up
    x_axis = np.cross(y_axis, optical_axis)
    return np.column_stack([x_axis, y_axis, optical_axis])


def attitude_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rotation_from_attitude` (returns roll, tilt_x, tilt_y)."""
    r = np.asarray(rotation, dtype=np.float64)
    optical_axis = r[:, 2]
    tilt_x = optical_axis[0] / optical_axis[2]
    tilt_y = optical_axis[1] / optical_axis[2]

    up0 = ENU_NORTH - np.dot(ENU_NORTH, optical_axis) * optical_axis
    if np.linalg.norm(up0) < 1e-6:
        up0 = ENU_EAST - np.dot(ENU_EAST, optical_axis) * optical_axis
    up0 = _unit(up0)
    # Column 1 is the camera's y axis, which points *down* the raster.
    up = -r[:, 1]
    roll = math.atan2(np.dot(np.cross(up0, up), optical_axis), np.dot(up0, up))
    return roll, tilt_x, tilt_y


@dataclass(frozen=True)
class CameraModel:
    """Lens + image centre + attitude: the complete pixel <-> sky mapping.

    ``xc`` and ``yc`` are the optical centre expressed as fractions of the frame
    height (both about 0.5); everything else is in radians.
    """

    lens: LensModel
    xc: float = 0.5
    yc: float = 0.5
    roll: float = 0.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    mirrored: bool = False

    # ------------------------------------------------------------- attitude
    def rotation(self) -> np.ndarray:
        """Camera-to-ENU rotation matrix."""
        return rotation_from_attitude(self.roll, self.tilt_x, self.tilt_y)

    def altaz_from_enu(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Altitude/azimuth of ENU vectors."""
        return altaz_from_enu(direction)

    def with_updates(self, **changes) -> "CameraModel":
        """Copy with selected fields replaced (keeps the dataclass immutable)."""
        return CameraModel(
            lens=changes.get("lens", self.lens),
            xc=float(changes.get("xc", self.xc)),
            yc=float(changes.get("yc", self.yc)),
            roll=float(changes.get("roll", self.roll)),
            tilt_x=float(changes.get("tilt_x", self.tilt_x)),
            tilt_y=float(changes.get("tilt_y", self.tilt_y)),
            mirrored=bool(changes.get("mirrored", self.mirrored)),
        )

    def ideal_equiangular(self) -> "CameraModel":
        """The same camera through an *ideal* equiangular lens.

        Same optical centre, same attitude, same mirror: only the radial law is
        replaced, by the equiangular curve through the rim this camera measured
        (:meth:`LensModel.equiangular`), so the image circle keeps its radius and
        its rim angle and the correction is a pure radial stretch about the axis.
        It is the ``mode='optical'`` output of :meth:`undistort_maps` stated as a
        camera: what the deliverable resamples into, before any attitude.
        A frame starved of pins gets the seed lens's ideal, which is still the
        right shape at the right scale -- it is just not a measurement.
        """
        return self.with_updates(
            lens=LensModel.equiangular(self.lens.theta_edge_deg, self.lens.sigma_edge)
        )

    # --------------------------------------------------- pixel <-> direction
    def centre_px(self, height_px: float) -> tuple[float, float]:
        """Optical centre in pixels of a frame with the given height."""
        return self.xc * height_px, self.yc * height_px

    def sigma_from_pixel(self, x, y, height_px: float) -> np.ndarray:
        """Normalised radius of pixel position(s)."""
        u, v = self.plane_from_pixel(x, y, height_px)
        return np.hypot(u, v)

    def plane_from_pixel(self, x, y, height_px: float):
        """Dimensionless image-plane coordinates ``(u, v)``, v pointing up."""
        u = (np.asarray(x, dtype=np.float64) / height_px - self.xc) * 2.0
        v = (self.yc - np.asarray(y, dtype=np.float64) / height_px) * 2.0
        if self.mirrored:
            u = -u
        return u, v

    def pixel_from_plane(self, u, v, height_px: float):
        """Inverse of :meth:`plane_from_pixel`."""
        u = np.asarray(u, dtype=np.float64)
        if self.mirrored:
            u = -u
        half = height_px / 2.0
        x = self.xc * height_px + u * half
        y = self.yc * height_px - np.asarray(v, dtype=np.float64) * half
        return x, y

    def camera_vectors(self, x, y, height_px: float) -> np.ndarray:
        """Camera-frame unit vector(s) of pixel position(s), attitude independent.

        The camera frame is right handed with y pointing *down* the raster, so
        the azimuth has to be measured with ``-v``: ``v`` points up in the image
        while the frame's y axis does not.
        """
        u, v = self.plane_from_pixel(x, y, height_px)
        sigma = np.hypot(u, v)
        phi = np.arctan2(-v, u)
        theta = self.lens.theta(sigma)
        return np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        )

    def enu_from_pixel(self, x, y, height_px: float) -> np.ndarray:
        """Unit vector(s) in ENU looking at pixel position(s) ``(x, y)``."""
        return self.camera_vectors(x, y, height_px) @ self.rotation().T

    def pixel_from_enu(self, direction: np.ndarray, height_px: float):
        """Pixel position(s) of ENU direction(s)."""
        d_cam = np.asarray(direction, dtype=np.float64) @ self.rotation()
        theta = np.arccos(np.clip(d_cam[..., 2], -1.0, 1.0))
        phi = np.arctan2(d_cam[..., 1], d_cam[..., 0])
        sigma = self.lens.sigma(theta)
        # Back from the y-down camera frame to the (v up) image plane.
        return self.pixel_from_plane(sigma * np.cos(phi), -sigma * np.sin(phi), height_px)

    def pixel_from_altaz(self, alt_rad, az_rad, height_px: float):
        """Pixel position(s) of altitude/azimuth in radians."""
        return self.pixel_from_enu(enu_from_altaz(alt_rad, az_rad), height_px)

    def altaz_from_pixel(self, x, y, height_px: float):
        """Altitude/azimuth (radians) of pixel position(s)."""
        return altaz_from_enu(self.enu_from_pixel(x, y, height_px))

    # ------------------------------------------------------ equiangular output
    def output_scale(self, out_size: int) -> float:
        """Pixels per radian of an equiangular output raster of ``out_size``.

        The undistorted disc keeps the image circle's relative size, so the
        plate scale is fixed to the input ("same as input"): the circle edge
        lands on the frame edge and spans ``lens.theta_edge`` radians.
        """
        return equiangular_pixels_per_radian(self.lens, float(out_size))

    def undistorted_from_pixel(self, x, y, height_px: float, out_size: int, mode: str = "optical"):
        """Map input pixel position(s) into the equiangular output raster.

        ``mode='optical'`` keeps the disc centred on the optical axis (a pure
        radial stretch -- used for the calibration preview); ``mode='zenith'``
        centres the disc on the zenith, so altitude/azimuth read directly off
        the output (used for the final deliverable).
        """
        centre = out_size / 2.0
        f_out = self.output_scale(out_size)
        if mode == "optical":
            u, v = self.plane_from_pixel(x, y, height_px)
            theta = self.lens.theta(np.hypot(u, v))
            phi = np.arctan2(v, u)
            return centre + f_out * theta * np.cos(phi), centre - f_out * theta * np.sin(phi)

        alt, az = self.altaz_from_pixel(x, y, height_px)
        radius = (0.5 * np.pi - alt) * f_out
        return centre + radius * np.sin(az), centre - radius * np.cos(az)

    def undistort_maps(self, height_px: float, out_size: int, mode: str = "optical"):
        """Build ``cv2.remap`` maps for the equiangular output raster.

        Returns ``(map_x, map_y, valid)`` where ``valid`` marks output pixels
        that fall inside the image circle.  The maps are float32, ready for
        :func:`cv2.remap`.
        """
        coords = np.arange(out_size, dtype=np.float64) + 0.5
        nx, ny = np.meshgrid(
            (coords - out_size / 2.0) / (out_size / 2.0),
            (out_size / 2.0 - coords) / (out_size / 2.0),
        )
        rho = np.hypot(nx, ny)
        # Output raster is equiangular and "same size as input": the image
        # circle edge (rho == sigma_edge) keeps its radius and spans theta_edge.
        theta = rho * (self.lens.theta_edge / self.lens.sigma_edge)

        if mode == "optical":
            # Azimuth is preserved, so the map is a pure radial stretch; the
            # radial part comes from a cached 1-D lookup table (much cheaper
            # than evaluating the inverse polynomial at every output pixel).
            safe_rho = np.maximum(rho, 1e-12)
            cos_phi = np.where(rho > 0.0, nx / safe_rho, 1.0)
            sin_phi = np.where(rho > 0.0, ny / safe_rho, 0.0)
            radius_px = np.interp(rho, *radial_lut(self.lens, height_px, float(rho.max())))
            map_x, map_y = self.pixel_from_plane(
                radius_px * cos_phi / (height_px / 2.0),
                radius_px * sin_phi / (height_px / 2.0),
                height_px,
            )

        elif mode == "zenith":
            alt = 0.5 * np.pi - theta
            d_enu = enu_from_altaz(alt, np.arctan2(nx, ny))
            map_x, map_y = self.pixel_from_enu(d_enu, height_px)
        else:
            raise ValueError(f"unknown undistort mode {mode!r}")

        valid = rho <= 1.0
        return map_x.astype(np.float32), map_y.astype(np.float32), valid


    def remap_equiangular(
        self,
        image: np.ndarray,
        height_px: float,
        out_size: int,
        mode: str = "optical",
        flags: int = cv2.INTER_CUBIC,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample ``image`` into equiangular projection.

        Returns the resampled image (black outside the circle) and the
        coverage mask.  The caller decides the raster size, so previews and the
        full-resolution deliverable share one code path.
        """
        map_x, map_y, valid = self.undistort_maps(height_px, out_size, mode)
        warped = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped[~valid] = 0
        return warped, valid

    # ----------------------------------------------------------- reporting
    def arcmin_per_pixel(self, height_px: float, sigma: float = 0.0) -> float:
        """Local angular scale (arcmin per pixel) at normalised radius ``sigma``."""
        slope = float(self.lens.dtheta_dsigma(np.array([sigma]))[0])
        return math.degrees(slope) * 60.0 / (height_px / 2.0)

    # -------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        """JSON-ready representation of lens + centre + attitude."""
        return {
            "lens": self.lens.to_dict(),
            "centre": {"xc_frac": self.xc, "yc_frac": self.yc},
            "attitude": {
                "roll_rad": self.roll,
                "tilt_x_rad": self.tilt_x,
                "tilt_y_rad": self.tilt_y,
                "roll_deg": math.degrees(self.roll),
                "tilt_x_deg": math.degrees(self.tilt_x),
                "tilt_y_deg": math.degrees(self.tilt_y),
            },
            "mirrored": self.mirrored,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraModel":
        """Rebuild a camera model from :meth:`to_dict` output."""
        centre = payload.get("centre", {})
        attitude = payload.get("attitude", {})
        return cls(
            lens=LensModel.from_dict(payload["lens"]),
            xc=float(centre.get("xc_frac", 0.5)),
            yc=float(centre.get("yc_frac", 0.5)),
            roll=float(attitude.get("roll_rad", 0.0)),
            tilt_x=float(attitude.get("tilt_x_rad", 0.0)),
            tilt_y=float(attitude.get("tilt_y_rad", 0.0)),
            mirrored=bool(payload.get("mirrored", False)),
        )


# ------------------------------------------------------------ correction maps
def remap_to_ideal(
    image: np.ndarray,
    camera: "CameraModel",
    ideal: "CameraModel",
    block_rows: int = 512,
    flags: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Resample ``image`` so the lens reads as ``ideal``, at the raster's own size.

    ``ideal`` is normally the camera's own equiangular disc
    (:meth:`CameraModel.ideal_equiangular`), and this is the same resampling
    :meth:`CameraModel.undistort_maps` builds for ``mode='optical'`` -- the disc
    keeps its size, so output radius fraction and input radius fraction mean the
    same field angle.

    Output pixel -> its radius about the ideal axis -> the field angle the ideal
    lens gives that radius -> the radius ``camera`` puts the *same* angle at ->
    the photograph pixel to sample.  Azimuth is untouched, so the whole map is
    one radius table (:func:`ideal_radius_lut`) plus a per-pixel scale, and both
    cameras are anchored at the same rim: the image circle keeps its radius, its
    centre and its orientation, and only the stretch inside it changes.  That is
    what makes a flash comparison against the photograph show the *distortion*
    moving rather than the picture jumping about.

    Everything outside the circle is black, as in the photograph.  The maps are
    built a band of rows at a time and handed straight to :func:`cv2.remap`, so a
    full-resolution frame costs one output array and one band of float32 maps
    instead of four image-sized grids.
    """
    height, width = image.shape[:2]
    centre_x, centre_y = camera.centre_px(float(height))
    ideal_x, ideal_y = ideal.centre_px(float(height))
    radius_out, radius_src = ideal_radius_lut(camera, ideal, float(height))
    rim = float(radius_out[-1])

    out = np.zeros_like(image)
    columns = np.arange(width, dtype=np.float64) + 0.5
    for first in range(0, height, int(block_rows)):
        stop = min(height, first + int(block_rows))
        rows = np.arange(first, stop, dtype=np.float64) + 0.5
        dx = (columns - ideal_x)[None, :]
        dy = (rows - ideal_y)[:, None]
        radius = np.hypot(dx, dy)
        inside = radius <= rim
        # The output radius -> photograph radius table, clamped at 0 on the axis
        # where the direction of the stretch is not defined anyway.
        wanted = np.where(inside, np.interp(radius, radius_out, radius_src), 0.0)
        stretch = np.divide(wanted, radius, out=np.zeros_like(radius), where=radius > 0.0)
        map_x = (centre_x + dx * stretch).astype(np.float32)
        map_y = (centre_y + dy * stretch).astype(np.float32)
        band = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        band[~inside] = 0
        out[first:stop] = band
    LOGGER.debug(
        "resampled %dx%d onto an ideal equiangular lens (rim %.1f px)",
        width, height, rim,
    )
    return out

