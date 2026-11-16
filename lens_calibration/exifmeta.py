"""EXIF time / position extraction for calibration frames.

The calibration needs three things from the file: which way the camera was
pointing (which the pins measure), *when* the shutter opened, and *where* it
was.  Sony bodies write the capture instant as local time plus an optional UTC
offset, and GPS fields carry their own UTC stamp.  Everything is returned
together with a provenance string so the UI can flag assumptions instead of
silently trusting a wrong clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from PIL import ExifTags, Image

LOGGER = logging.getLogger(__name__)

# EXIF tag ids used below (kept local so the parsing code has no magic numbers).
TAG_DATETIME_ORIGINAL = 36867
TAG_OFFSET_TIME_ORIGINAL = 36881
TAG_OFFSET_TIME = 36880
TAG_DATETIME = 306
TAG_MAKE = 271
TAG_MODEL = 272
TAG_LENS_MODEL = 42036
TAG_F_NUMBER = 33437
TAG_EXPOSURE_TIME = 33434
TAG_ISO = 34855
TAG_FOCAL_LENGTH = 37386

GPS_LAT_REF = 1
GPS_LAT = 2
GPS_LON_REF = 3
GPS_LON = 4
GPS_ALT_REF = 5
GPS_ALT = 6
GPS_TIME = 7
GPS_DATE = 29


@dataclass(frozen=True)
class ExposureMeta:
    """Metadata of one frame (every optional field may be ``None``)."""

    path: str
    width: int
    height: int
    captured_local: datetime | None = None
    utc_offset: timedelta | None = None
    gps_utc: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    camera: str | None = None
    lens: str | None = None
    exposure_s: float | None = None
    iso: int | None = None
    f_number: float | None = None
    focal_length_mm: float | None = None

    # --------------------------------------------------------------- derived
    @property
    def captured_utc(self) -> datetime | None:
        """Best-effort UTC capture instant (see :attr:`time_source`)."""
        if self.captured_local is not None and self.utc_offset is not None:
            return (self.captured_local - self.utc_offset).replace(tzinfo=timezone.utc)
        if self.gps_utc is not None:
            return self.gps_utc
        if self.captured_local is not None:
            # Sony writes the camera clock, normally set to local time.  Without
            # an offset we treat it as UTC and let the UI correct it.
            return self.captured_local.replace(tzinfo=timezone.utc)
        return None

    @property
    def time_source(self) -> str:
        """Human readable provenance of :attr:`captured_utc`."""
        if self.captured_local is not None and self.utc_offset is not None:
            return "exif DateTimeOriginal + OffsetTimeOriginal"
        if self.gps_utc is not None:
            return "exif GPS date/time stamp"
        if self.captured_local is not None:
            return "exif DateTimeOriginal, timezone ASSUMED UTC (correct me)"
        return "no time in file (enter manually)"

    @property
    def has_position(self) -> bool:
        """True when latitude and longitude are both known."""
        return self.latitude is not None and self.longitude is not None

    @property
    def position_source(self) -> str:
        """Human readable provenance of the observer position."""
        return "exif GPS" if self.has_position else "not in file (enter manually)"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary for the UI."""
        utc = self.captured_utc
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "captured_local": self.captured_local.isoformat() if self.captured_local else None,
            "captured_utc": utc.isoformat() if utc else None,
            "utc_offset_minutes": (
                int(self.utc_offset.total_seconds() / 60) if self.utc_offset else None
            ),
            "time_source": self.time_source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "position_source": self.position_source,
            "camera": self.camera,
            "lens": self.lens,
            "exposure_s": self.exposure_s,
            "iso": self.iso,
            "f_number": self.f_number,
            "focal_length_mm": self.focal_length_mm,
        }


def _rational(value: Any) -> float | None:
    """Coerce an EXIF rational (tuple, IFDRational or number) to float."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator) if denominator else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_degrees(value: Any, ref: Any) -> float | None:
    """Convert a GPS ``(deg, min, sec)`` sequence to signed decimal degrees."""
    if value is None:
        return None
    try:
        parts = [_rational(part) or 0.0 for part in value]
    except TypeError:
        return None
    if len(parts) != 3:
        return None
    degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if isinstance(ref, str) and ref.strip().upper() in {"S", "W"}:
        degrees = -degrees
    return degrees


def _parse_exif_datetime(text: Any) -> datetime | None:
    """Parse ``"YYYY:MM:DD HH:MM:SS"`` into a naive datetime."""
    if isinstance(text, bytes):
        text = text.decode("ascii", "ignore")
    if not isinstance(text, str):
        return None
    text = text.strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    LOGGER.debug("unparsable EXIF datetime %r", text)
    return None


def _parse_offset(text: Any) -> timedelta | None:
    """Parse ``"+02:00"`` style EXIF UTC offsets."""
    if isinstance(text, bytes):
        text = text.decode("ascii", "ignore")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) != 6 or text[3] != ":" or text[0] not in "+-":
        return None
    try:
        sign = 1 if text[0] == "+" else -1
        hours, minutes = int(text[1:3]), int(text[4:6])
    except ValueError:
        return None
    return sign * timedelta(hours=hours, minutes=minutes)


def read_exposure_meta(path: str) -> ExposureMeta:
    """Read size, time and position metadata from an image file.

    Missing tags never raise: absent values stay ``None`` and the UI asks for
    them.  An unreadable file still raises -- there is nothing to calibrate.
    """
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()

    gps: dict[int, Any] = {}
    try:
        gps = dict(exif.get_ifd(ExifTags.IFD.GPSInfo))
    except (AttributeError, KeyError, TypeError, OSError):
        LOGGER.debug("no GPS IFD in %s", path)

    captured_local = _parse_exif_datetime(
        exif.get(TAG_DATETIME_ORIGINAL) or exif.get(TAG_DATETIME)
    )
    utc_offset = _parse_offset(exif.get(TAG_OFFSET_TIME_ORIGINAL)) or _parse_offset(
        exif.get(TAG_OFFSET_TIME)
    )

    gps_utc = None
    gps_date = gps.get(GPS_DATE)
    gps_time = gps.get(GPS_TIME)
    if isinstance(gps_date, (str, bytes)) and gps_time is not None:
        try:
            if isinstance(gps_date, bytes):
                gps_date = gps_date.decode("ascii", "ignore")
            stamp = [float(_rational(part)) for part in gps_time]
            day = datetime.strptime(gps_date.strip().replace("\x00", ""), "%Y:%m:%d")
            gps_utc = (day + timedelta(hours=stamp[0], minutes=stamp[1], seconds=stamp[2])).replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError, IndexError):
            LOGGER.debug("unparsable GPS timestamp in %s", path)

    latitude = _dms_to_degrees(gps.get(GPS_LAT), gps.get(GPS_LAT_REF))
    longitude = _dms_to_degrees(gps.get(GPS_LON), gps.get(GPS_LON_REF))
    altitude = _rational(gps.get(GPS_ALT))
    if altitude is not None and gps.get(GPS_ALT_REF) in (1, b"\x01"):
        altitude = -altitude

    make = str(exif.get(TAG_MAKE) or "").strip() or None
    model = str(exif.get(TAG_MODEL) or "").strip() or None
    iso_values = exif.get(TAG_ISO)
    iso = None
    if isinstance(iso_values, (tuple, list)) and iso_values:
        iso = int(iso_values[0])
    elif iso_values is not None:
        iso = int(iso_values)

    return ExposureMeta(
        path=str(path),
        width=int(width),
        height=int(height),
        captured_local=captured_local,
        utc_offset=utc_offset,
        gps_utc=gps_utc,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude,
        camera=" ".join(part for part in (make, model) if part) or None,
        lens=str(exif.get(TAG_LENS_MODEL) or "").strip() or None,
        exposure_s=_rational(exif.get(TAG_EXPOSURE_TIME)),
        iso=iso,
        f_number=_rational(exif.get(TAG_F_NUMBER)),
        focal_length_mm=_rational(exif.get(TAG_FOCAL_LENGTH)),
    )

