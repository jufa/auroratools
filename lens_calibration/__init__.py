"""Lens calibration tool for a circular fisheye (Meike 6.5 mm on a Sony ZV-E10).

The package fits a radial projection model ``theta(sigma)`` -- field angle as a
function of normalised image radius -- against a bright-star catalogue, for a
single exposure whose time and position are known from EXIF.  All geometry is
kept dimensionless (see :mod:`lens_calibration.model`) so one calibration file
describes every raster size of the same field of view.

Modules
-------
model      radial lens model + JSON (de)serialisation
geometry   camera attitude, pixel <-> sky conversions, equiangular resampling
exifmeta   EXIF time / position extraction
catalog    bundled bright-star catalogue and alt/az prediction
aids       draw-only landmarks: clusters, nebulae, the Milky Way band
detect     star detection and centroiding on a single frame
match      predicted-to-detected star association
selfcheck  geometry invariants and a synthetic solve
           (``python -m lens_calibration.selfcheck``)
fit        attitude + lens least squares (alternating closed form, then joint)
session    per-image state shared by the solver and the web UI
server     Flask + Socket.IO application (the interactive calibration UI)
"""

__version__ = "1.0.0"
