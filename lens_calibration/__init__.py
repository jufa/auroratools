"""Lens calibration tool for a circular fisheye (Meike 6.5 mm on a Sony ZV-E10).

The package fits a radial projection model ``theta(sigma)`` -- field angle as a
function of normalised image radius -- against the stars the user pins by hand
on one exposure whose time and position are known from EXIF.  All geometry is
kept dimensionless (see :mod:`lens_calibration.model`) so one calibration file
describes every raster size of the same field of view.

Modules
-------
model      radial lens model + JSON (de)serialisation
geometry   camera attitude, pixel <-> sky conversions, equiangular resampling
exifmeta   EXIF time / position extraction
catalog    bundled bright-star catalogue and alt/az prediction
aids       draw-only landmarks: clusters, nebulae, the Milky Way band
match      image-circle fit and the closed-form attitude update
session    per-image state: the overlay, the manual alignment, the pin fit
server     Flask application (the interactive calibration UI)
selfcheck  geometry invariants, pin fit and page checks
           (``python -m lens_calibration.selfcheck``)
"""

__version__ = "2.0.0"
