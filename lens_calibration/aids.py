"""Visual alignment aids: star clusters, nebulae and the Milky Way.

These are drawn, never fitted.  A pin has to hang off something with a single
position -- a catalogue star, which the catalogue predicts to an arcsecond and
therefore fit against -- whereas a cluster is a fuzzy patch with no centre worth
clicking.  What the aids answer is the question the rings cannot: "is that bright
smudge the Pleiades, or the Double Cluster?", while the photo is still being
dragged into place.  They are therefore drawn in their own colour, they are not
in the overlay's ``stars`` list, and nothing in the pin path of
:mod:`lens_calibration.session` can see them.

The Milky Way is a band rather than an object: it is sampled along the galactic
plane and given a schematic half-width that widens towards the galactic centre.
That width is a drawing cue, not photometry -- the real band is patchy, and
modelling the patches would be a texture, not an alignment aid.

Positions are ICRS (J2000) centres and approximate angular radii, rounded to the
point where they help someone recognise a smudge on a stretched preview; they
are not a substitute for a proper deep-sky catalogue.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils import iers
from erfa import ErfaWarning

from .catalog import observer_frame

# The catalogue module switches the IERS bulletin off for the whole process (a
# frame must never wait on the network); repeating it here keeps this module
# usable on its own, e.g. from the self checks.
iers.conf.auto_download = False

#: What an aid is, for the page's colours and labels: a cluster is a swarm of
#: stars, a galaxy and a nebula are diffuse, a dark cloud is a hole in the band.
AID_KINDS = ("cluster", "galaxy", "nebula", "dark")


@dataclass(frozen=True)
class SkyAid:
    """One draw-only object: where it is and how big it looks."""

    name: str
    kind: str
    ra_deg: float
    dec_deg: float
    radius_deg: float


#: The aids worth recognising on a 180 deg fisheye.  Both hemispheres: a frame
#: shot from Alaska and one shot from New Zealand have to have landmarks.
#: ``radius_deg`` is the visual extent, not a core radius, so the circle drawn
#: for the LMC (5.4 deg) is the size of the whole galaxy.
SKY_AIDS: tuple[SkyAid, ...] = (
    # --- open and globular clusters ------------------------------------------
    SkyAid("Pleiades (M45)", "cluster", 56.75, 24.117, 1.8),
    SkyAid("Hyades", "cluster", 66.75, 15.87, 3.0),
    SkyAid("Alpha Persei", "cluster", 51.0, 49.0, 1.8),
    SkyAid("Double Cluster", "cluster", 34.74, 57.13, 0.6),
    SkyAid("Beehive (M44)", "cluster", 130.1, 19.67, 0.8),
    SkyAid("Coma Berenices", "cluster", 186.0, 26.0, 2.5),
    SkyAid("Southern Pleiades", "cluster", 160.75, -64.4, 0.9),
    SkyAid("Jewel Box", "cluster", 193.4, -60.4, 0.15),
    SkyAid("Omega Centauri", "cluster", 201.69, -47.48, 0.5),
    SkyAid("47 Tucanae", "cluster", 6.02, -72.08, 0.4),
    SkyAid("Ptolemy (M7)", "cluster", 268.46, -34.79, 0.7),
    SkyAid("Butterfly (M6)", "cluster", 265.07, -32.25, 0.4),
    SkyAid("Wild Duck (M11)", "cluster", 282.77, -6.27, 0.2),
    SkyAid("Hercules (M13)", "cluster", 250.42, 36.46, 0.2),
    SkyAid("Sagittarius (M22)", "cluster", 279.10, -23.90, 0.3),
    SkyAid("Sgr star cloud (M24)", "cluster", 274.2, -18.5, 0.5),
    # --- galaxies ------------------------------------------------------------
    SkyAid("LMC", "galaxy", 80.9, -69.75, 5.4),
    SkyAid("SMC", "galaxy", 13.19, -72.83, 2.5),
    SkyAid("Andromeda (M31)", "galaxy", 10.68, 41.27, 1.6),
    SkyAid("Triangulum (M33)", "galaxy", 23.46, 30.66, 0.6),
    # --- bright nebulae ------------------------------------------------------
    SkyAid("Orion Nebula (M42)", "nebula", 83.82, -5.39, 0.6),
    SkyAid("Barnard's Loop", "nebula", 84.0, -3.0, 5.0),
    SkyAid("North America", "nebula", 314.0, 44.3, 1.5),
    SkyAid("Veil", "nebula", 312.75, 30.7, 1.5),
    SkyAid("California", "nebula", 60.9, 36.4, 1.3),
    SkyAid("Rosette", "nebula", 97.9, 4.95, 0.7),
    SkyAid("Carina", "nebula", 161.3, -59.9, 1.5),
    SkyAid("Lagoon (M8)", "nebula", 270.9, -24.4, 0.5),
    SkyAid("Rho Ophiuchi", "nebula", 246.8, -24.5, 2.0),
    # --- dark clouds: a hole in the band, which is how they read on a photo ---
    SkyAid("Coalsack", "dark", 192.5, -63.0, 3.5),
)

#: Longitude step of the band polygon.  Fine enough that the band still bends
#: like a curve where the projection compresses it, coarse enough that the
#: polygon travels with every overlay response without swelling it.
BAND_LON_STEP_DEG = 2.5


def band_half_width_deg(galactic_lon_deg) -> np.ndarray:
    """Schematic half-width (deg) of the Milky Way at a galactic longitude.

    Widest at the bulge (``l = 0``, the Sagittarius star clouds that make the
    band unmistakable) with a second, smaller bump at the Cygnus star cloud
    (``l = 80``), and narrowest towards the anticentre.  Generous rather than
    accurate: the task is to recognise the band under a stretched preview.
    """
    lon = np.asarray(galactic_lon_deg, dtype=float)
    centre = np.abs((lon + 180.0) % 360.0 - 180.0)
    cygnus = np.abs((lon - 80.0 + 180.0) % 360.0 - 180.0)
    return 5.5 + 9.0 * np.exp(-np.square(centre / 35.0)) + 2.5 * np.exp(-np.square(cygnus / 25.0))


@dataclass(frozen=True)
class AidSky:
    """Every aid above the horizon at one instant, plus the band edges.

    ``band_alt``/``band_az`` are ``(2, n)``: the row at ``+b`` and the row at
    ``-b`` for the same longitudes, so a caller can zip them into a filled
    polygon without knowing anything about galactic coordinates.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    radius_deg: np.ndarray
    alt: np.ndarray
    az: np.ndarray
    band_lon_deg: np.ndarray
    band_half_width_deg: np.ndarray
    band_alt: np.ndarray
    band_az: np.ndarray

    def __len__(self) -> int:
        return len(self.names)



@lru_cache(maxsize=8)
def _sky_aids_cached(
    when_iso: str,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
    min_alt_deg: float,
) -> AidSky:
    when = datetime.fromisoformat(when_iso)
    altaz = observer_frame(latitude_deg, longitude_deg, when, elevation_m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        centres = SkyCoord(
            ra=np.array([aid.ra_deg for aid in SKY_AIDS]) * u.deg,
            dec=np.array([aid.dec_deg for aid in SKY_AIDS]) * u.deg,
            frame="icrs",
        ).transform_to(altaz)
        lon = np.arange(0.0, 360.0, BAND_LON_STEP_DEG)
        half = band_half_width_deg(lon)
        # Both edges in one transform: the band is a strip of fixed width in
        # galactic latitude, so the two rows differ only in the sign of b.
        edges = SkyCoord(
            l=np.concatenate([lon, lon]) * u.deg,
            b=np.concatenate([half, -half]) * u.deg,
            frame="galactic",
        ).transform_to(altaz)
    alt = centres.alt.to_value("rad")
    az = centres.az.to_value("rad")
    keep = alt >= math.radians(min_alt_deg)
    return AidSky(
        names=tuple(aid.name for aid, flag in zip(SKY_AIDS, keep) if flag),
        kinds=tuple(aid.kind for aid, flag in zip(SKY_AIDS, keep) if flag),
        radius_deg=np.radians([aid.radius_deg for aid in SKY_AIDS])[keep],
        alt=alt[keep],
        az=az[keep],
        band_lon_deg=lon,
        band_half_width_deg=np.radians(half),
        band_alt=edges.alt.to_value("rad").reshape(2, -1),
        band_az=edges.az.to_value("rad").reshape(2, -1),
    )


def sky_aids(
    when_utc: datetime,
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float = 0.0,
    min_alt_deg: float = 0.0,
) -> AidSky:
    """Aids and band above ``min_alt_deg`` at the given time and place.

    Same contract as :func:`lens_calibration.catalog.star_field`: a timezone
    aware UTC instant, and a cache keyed on the ISO form so a page that redraws
    on every drag does not re-run astropy on every request.
    """
    if when_utc.tzinfo is None:
        raise ValueError("sky_aids needs a timezone-aware UTC datetime")
    return _sky_aids_cached(
        when_utc.astimezone(timezone.utc).isoformat(),
        float(latitude_deg),
        float(longitude_deg),
        float(elevation_m),
        float(min_alt_deg),
    )
