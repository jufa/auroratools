"""Bright-star catalogue and alt/az predictions for a known time and place.

The catalogue is a trimmed Hipparcos extract (V <= 6.6, about 9k stars) with
proper motions and B-V colours, checked into ``data/star_catalog_bsc.csv`` so
calibration works offline.  ``python -m lens_calibration.catalog --refresh``
rebuilds it from VizieR (mirror ``vizier.cfa.harvard.edu``, because the CDS
mirror is unreachable from some networks) plus the IAU catalogue of star names.

Predictions use an optical (refraction-free) altitude: atmospheric refraction is
a smooth, low-frequency effect that the fitted radial polynomial absorbs, so
including a refraction model here would just shift the fitted field angles by a
few arcminutes at low altitude.
"""

from __future__ import annotations

import argparse
import argparse
import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers
from erfa import ErfaWarning

# Calibration must never block on the network for IERS bulletins; the bundled
# astropy-iers-data release gives ample accuracy for alt/az work.  The staleness
# limit goes with it: astropy raises when the predictive part of the table is
# older than ``auto_max_age``, which would make every frame shot after that
# window unusable (the overlay and the solver both read the field) on any
# machine that cannot download a fresh bulletin.  Predicted UT1-UTC is good to
# ~0.1 s, i.e. ~1.5 arcsec of sky rotation, far below anything this alignment
# cares about.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

LOGGER = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CATALOG_PATH = DATA_DIR / "star_catalog.csv"
CATALOG_VERSION = "hip_vmag6.6_j1991.25"

#: Hipparcos positions refer to this epoch (they are given with proper motions).
HIP_EPOCH = Time("J1991.25")

VIZIER_URL = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
VIZIER_CATALOG = "I/239/hip_main"
IAU_NAMES_URL = "https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt"
MAG_LIMIT = 6.6

CATALOG_COLUMNS = ("hip", "name", "ra_deg", "dec_deg", "vmag", "bv", "pmra_mas_yr", "pmdec_mas_yr")


@dataclass(frozen=True)
class StarField:
    """Catalogue stars above the horizon at one instant and place."""

    names: tuple[str, ...]
    hip: np.ndarray
    vmag: np.ndarray
    bv: np.ndarray
    alt: np.ndarray
    az: np.ndarray
    enu: np.ndarray

    def __len__(self) -> int:
        return int(self.vmag.size)

    def select(self, max_mag: float) -> "StarField":
        """Subset brighter than ``max_mag``."""
        keep = self.vmag <= max_mag
        return StarField(
            names=tuple(name for name, flag in zip(self.names, keep) if flag),
            hip=self.hip[keep],
            vmag=self.vmag[keep],
            bv=self.bv[keep],
            alt=self.alt[keep],
            az=self.az[keep],
            enu=self.enu[keep],
        )


# --------------------------------------------------------------------- loading
@lru_cache(maxsize=4)
def load_catalog(path: str = str(CATALOG_PATH)) -> pd.DataFrame:
    """Load the bundled catalogue (cached; raises if it is missing)."""
    frame = pd.read_csv(path, dtype={"name": "string"})
    missing = set(CATALOG_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return frame


@lru_cache(maxsize=8)
def _catalog_skycoord(path: str = str(CATALOG_PATH)) -> SkyCoord:
    """Catalogue positions as a proper-motion aware ``SkyCoord``."""
    frame = load_catalog(path)
    return SkyCoord(
        ra=frame["ra_deg"].to_numpy() * u.deg,
        dec=frame["dec_deg"].to_numpy() * u.deg,
        pm_ra_cosdec=frame["pmra_mas_yr"].fillna(0.0).to_numpy() * u.mas / u.yr,
        pm_dec=frame["pmdec_mas_yr"].fillna(0.0).to_numpy() * u.mas / u.yr,
        obstime=HIP_EPOCH,
        frame="icrs",
    )


def observer_frame(latitude_deg: float, longitude_deg: float, when_utc: datetime,
                   elevation_m: float = 0.0) -> AltAz:
    """Topocentric ``AltAz`` frame with refraction switched off."""
    location = EarthLocation(
        lat=latitude_deg * 1.0,  # astropy accepts degrees for EarthLocation
        lon=longitude_deg * 1.0,
        height=elevation_m,
    )
    return AltAz(obstime=Time(when_utc), location=location, pressure=0)


@lru_cache(maxsize=32)
def _star_field_cached(
    when_iso: str,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
    min_alt_deg: float,
    catalog_path: str,
) -> StarField:
    frame = load_catalog(catalog_path)
    when = datetime.fromisoformat(when_iso)
    altaz = observer_frame(latitude_deg, longitude_deg, when, elevation_m)

    # Proper motions are applied on the unit sphere; the resulting velocities
    # and the nominal distance are dropped because alt/az needs directions only
    # (keeping them makes astropy demand a real distance for the cartesian step).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        moved = _catalog_skycoord(catalog_path).apply_space_motion(new_obstime=Time(when))
    horizontal = SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs").transform_to(altaz)

    alt = horizontal.alt.to_value("rad")
    az = horizontal.az.to_value("rad")
    keep = alt >= math.radians(min_alt_deg)
    names = tuple(
        (name if isinstance(name, str) and name else f"HIP {int(hip)}")
        for name, hip in zip(frame["name"].fillna("").to_numpy(), frame["hip"].to_numpy())
    )
    return StarField(
        names=tuple(name for name, flag in zip(names, keep) if flag),
        hip=frame["hip"].to_numpy()[keep],
        vmag=frame["vmag"].to_numpy()[keep],
        bv=frame["bv"].fillna(0.6).to_numpy()[keep],
        alt=alt[keep],
        az=az[keep],
        enu=np.stack(
            [
                np.cos(alt[keep]) * np.sin(az[keep]),
                np.cos(alt[keep]) * np.cos(az[keep]),
                np.sin(alt[keep]),
            ],
            axis=-1,
        ),
    )


def star_field(
    when_utc: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float = 0.0,
    min_alt_deg: float = 0.0,
    catalog_path: str | Path = CATALOG_PATH,
) -> StarField:
    """All catalogue stars above ``min_alt_deg`` at the given time/place."""
    if when_utc.tzinfo is None:
        raise ValueError("star_field needs a timezone-aware UTC datetime")
    return _star_field_cached(
        when_utc.astimezone(timezone.utc).isoformat(),
        float(latitude_deg),
        float(longitude_deg),
        float(elevation_m),
        float(min_alt_deg),
        str(catalog_path),
    )


# ---------------------------------------------------------------- solar system
def sun_altitude_deg(when_utc: datetime, latitude_deg: float, longitude_deg: float,
                     elevation_m: float = 0.0) -> float:
    """Apparent (unrefracted) solar altitude -- the sanity check for the clock."""
    altaz = observer_frame(latitude_deg, longitude_deg, when_utc, elevation_m)
    return float(get_sun(Time(when_utc)).transform_to(altaz).alt.to_value("deg"))


def solar_system_bodies(
    when_utc: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float = 0.0,
    bodies: tuple[str, ...] = ("moon", "mercury", "venus", "mars", "jupiter", "saturn"),
) -> dict[str, dict[str, float]]:
    """Alt/az (deg) and angular radius (deg) of bright solar system bodies.

    Used to exclude stars that share a centroid with a planet or the Moon.
    """
    altaz = observer_frame(latitude_deg, longitude_deg, when_utc, elevation_m)
    result: dict[str, dict[str, float]] = {}
    for body in bodies:
        try:
            horizontal = get_body(body, Time(when_utc), location=altaz.location).transform_to(altaz)
        except Exception as error:  # noqa: BLE001 - ephemeris gaps must not kill a solve
            LOGGER.warning("could not compute %s position: %s", body, error)
            continue
        try:
            radius_deg = horizontal.angular_radius.to_value("deg")
        except Exception:  # noqa: BLE001 - planets have no angular radius attribute
            radius_deg = 0.0
        result[body] = {
            "alt_deg": float(horizontal.alt.to_value("deg")),
            "az_deg": float(horizontal.az.to_value("deg")),
            "angular_radius_deg": float(radius_deg),
        }
    return result


# ------------------------------------------------------------- catalogue build
def _fetch_hipparcos(timeout: float = 180.0) -> pd.DataFrame:
    """Download the Hipparcos main catalogue fields we need from VizieR."""
    import requests  # local import: only needed when (re)building the catalogue

    params = {
        "-source": VIZIER_CATALOG,
        "-out": "HIP,RAICRS,DEICRS,Vmag,pmRA,pmDE,B-V",
        "-out.max": "unlimited",
    }
    response = requests.get(VIZIER_URL, params=params, timeout=timeout)
    response.raise_for_status()

    rows = response.text.splitlines()
    header_index = next(
        (index for index, line in enumerate(rows) if line.startswith("HIP\t")), None
    )
    if header_index is None:
        raise RuntimeError("unexpected VizieR response: no HIP header found")

    records: list[tuple] = []
    for line in rows[header_index + 3:]:
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        try:
            hip = int(fields[0])
            ra_deg = float(fields[1])
            dec_deg = float(fields[2])
            vmag = float(fields[3])
        except ValueError:
            continue
        pm_ra = float(fields[4]) if fields[4].strip() else 0.0
        pm_dec = float(fields[5]) if fields[5].strip() else 0.0
        bv = float(fields[6]) if fields[6].strip() else np.nan
        records.append((hip, ra_deg, dec_deg, vmag, bv, pm_ra, pm_dec))

    LOGGER.info("fetched %d Hipparcos rows", len(records))
    return pd.DataFrame(records, columns=CATALOG_COLUMNS[0:1] + CATALOG_COLUMNS[2:])


def _fetch_iau_names(timeout: float = 60.0) -> dict[int, str]:
    """Download the IAU catalogue of star names, keyed by HIP number."""
    import re

    import requests

    response = requests.get(IAU_NAMES_URL, timeout=timeout)
    response.raise_for_status()
    pattern = re.compile(
        r"\s(\d{1,6})\s+(\d+)\s+(\d{1,3}\.\d{6})\s+([-+]?\d{1,2}\.\d{6})\s+\d{4}-\d{2}-\d{2}"
    )
    names: dict[int, str] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.search(line)
        if match is None:
            continue
        token = line.split()[0]
        if token in {"_", ""}:
            continue
        names[int(match.group(1))] = token
    LOGGER.info("fetched %d IAU star names", len(names))
    return names


def build_catalog(
    output_path: str | Path = CATALOG_PATH,
    mag_limit: float = MAG_LIMIT,
) -> Path:
    """(Re)build the bundled catalogue CSV from VizieR and the IAU name list."""
    frame = _fetch_hipparcos()
    names = _fetch_iau_names()

    frame = frame[frame["vmag"] <= mag_limit].copy()
    frame["name"] = frame["hip"].map(names).fillna("")
    frame = frame.sort_values("hip").reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        index=False,
        columns=list(CATALOG_COLUMNS),
        float_format="%.6f",
    )
    LOGGER.info("wrote %d stars brighter than %.1f to %s", len(frame), mag_limit, output)
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m lens_calibration.catalog --refresh``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="download and rewrite the catalogue")
    parser.add_argument("--output", default=str(CATALOG_PATH), help="catalogue CSV path")
    parser.add_argument("--mag-limit", type=float, default=MAG_LIMIT, help="faintest V magnitude")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.refresh:
        frame = load_catalog(args.output)
        print(f"{len(frame)} stars in {args.output}")
        return 0
    written = build_catalog(args.output, args.mag_limit)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


