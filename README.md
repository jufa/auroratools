
# auroratools using the pipeline for AurorEye cameras using Sony ZVE-10:
python process_captured_sequence.py --path '/Volumes/T7 Shield/AurorEye/seq_2025-11-13T01-41-35' --gamma 1.2 --unit_text $'UNIT 15\nWINNIPEG,MB'

## Lens calibration (`lens_calibration/`)

```
.venv/bin/python3 -m lens_calibration.server sampleimages/00000687.jpg
.venv/bin/python3 -m lens_calibration.selfcheck     # geometry, pin fit, page
```

The overlay is the **camera the frame had**, drawn in the photograph's own
pixels: lens, optical centre, roll, the tilt of the optical axis and the mirror.
Before a star is pinned that camera is the seed the rig measures — focal length
in pixels per radian, odd `k` terms, optical centre (582.3 px/rad,
`k = -0.10455, -0.00080, 0`, centre 700.1 / 703.7 px of the 1400 px work raster)
— pointing at the zenith for the EXIF time and place, north up; after that it is
whatever the pins solved. Align the photo under it by hand if you like, then pin
stars: click a circle, then the same star in the photo. Every pin is refitted at
once and drawn as a line from the photo to its circle, so a pair that disagrees
is visible rather than averaged in.

- **Pins are named stars, not screen positions.** A pair is the star's name plus
the pixel you clicked in the photograph, so the answer cannot depend on how the
photo happens to be dragged — a frame can be turned after it has been pinned
without moving the calibration. Before the second pin the drag is the only
statement about where the sky is; afterwards it is just a way of looking at the
frame, and the rings ride along with the photo.
- **Each pin buys one degree of freedom.** One places the optical axis, two add
the roll, three also settle the *tilt* of the axis — which is what lets a frame
that does not point at the zenith be calibrated at all (a tilt stretches one side
of the image and compresses the other, and no shift or rotation can imitate it).
Three pins are also where a wrong mirror becomes decidable, and the fit corrects
it rather than trusting the toggle.
- **4 pins** are the minimum that measures the lens (optical centre and `k1`);
  the **5th** settles the `theta^5` term, the **6th** `theta^7` and the
  **7th** `theta^9` — and the fit stops there, because that is where
  `cv2.fisheye` stops too. Further stars are extra equations for the same
  unknowns, not more coefficients. Four pins leave only one spare equation for
  seven unknowns, so the tilt and `k1` can trade against each other; the readout
  says so.
- **Saving** writes the camera the pins measured — lens, optical centre, roll,
  tilt and mirror, in pixels of the photograph (`lens_calibration/2`) — to a JSON
  file **beside the photograph**, same stem (`sampleimages/00000687.jpg` →
  `sampleimages/00000687.json`), so a folder of frames carries its own
  calibrations. `--calibration-dir DIR` puts them in a folder instead, which is
  what a browser upload gets: the server only ever has its own copy under the
  temp directory, so those go to `./calibrations`. With fewer than four pins
  there is nothing to save: the file would hold the seed, not a measurement.
- **Everything is manual on purpose.** There is no star detector and no
  automatic solver: both were tried against real frames, both were worse than
  the person looking at the frame, and a wrong answer from either cost more than
  the click it saved.
- **Removing a pin**: click a circle that already carries one, or the ✕ beside
  its row in the readout. The survivors are refitted from scratch.
- **Sky aids** (checkbox "Sky aids") draw clusters, nebulae and the Milky Way
  band in violet as orientation cues. They are never fitted and cannot be
  pinned: a pin needs a single position, which only a catalogue star has.
- **The photo is shown as the camera wrote it.** `/frame.jpg` hands over the file
  at full resolution with no tone curve (byte for byte for a JPEG without an EXIF
  orientation, a lossless PNG otherwise), so the pixels a pin is placed on are
  the pixels the sensor recorded. A night frame is nearly black, so the *Frame
  view* card carries one **Gamma** slider: the exponent of a single SVG
  `feComponentTransfer` (`color-interpolation-filters="sRGB"`, one `feFunc` per
  channel) pointed at by the `<img>`'s `filter`. Below 1 lifts the shadows, 1.00
  sets no filter at all and *flat* returns to it. It is a view aid only — nothing
  is posted, nothing is cached, no pixel of the file changes, and the rings are
  drawn on their own canvas so they never pass through it. The card's **Zoom**
  sizes the stage against the photo's own pixels rather than the window: at 100 %
  the browser's own zoom has real detail to magnify instead of a downscaled copy.
- **View corrected image** swaps the picture for the same frame resampled onto an
  *ideal equiangular* lens (`/corrected.png`): full resolution, the photograph's
  own size and place, and the rings redrawn through that same ideal lens, so
  flipping between the two is a flash comparison of the measured distortion
  against the ideal one. Equiangular — the projection tables call the same law
  *equidistant*: `r = f·θ`, equal angles over equal radii — is the output the
  module already resamples into, so the preview is that resampling and nothing
  else. It is rendered when the button is pressed (the raster is
  a `cv2.remap` through one cached radius table, the answer a PNG the session
  keeps until the camera moves), it never touches the file on disk, and the pin
  side is frozen while it is up — a pin is a pixel of the photograph, and the
  correction has moved every one of them. `back to the photo` restores both.





