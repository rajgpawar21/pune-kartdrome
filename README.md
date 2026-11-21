# Pune Kartdrome: sketch and racing line

An etched-style vector sketch of Pune Kartdrome traced from a satellite image, with a minimum-curvature racing line through all 13 corners.

Open `index.html` in a browser (it is self-contained). Toggle Sketch/Satellite, racing line, centerline and corner numbers. Pick a corner to see turn-in, apex, exit and braking. To host it, enable GitHub Pages on this repo (Settings, Pages, deploy from branch, root).

## Results

- 13 corners: 8 right, 5 left
- Lap length about 645 m (derived from the assumed width, not measured)
- Corners are numbered in driving order from the start/finish on the long straight

## Assumptions

The image has no scale and nothing is known about the kart, so these are estimates. Replace them with real figures to rescale everything.

| Item | Value used | Affects |
|---|---|---|
| Track width | 7 m (about 59 px, 0.118 m per pixel) | Every distance and radius |
| Lap length | about 645 m, derived | Not measured |
| Kart top speed | 61 km/h (17 m/s) | Straight speeds |
| Cornering grip | 1.1 g (11 m/s²) | Corner speeds |
| Braking | 0.9 g (9 m/s²) | Brake zones |
| Acceleration | 4 m/s² | Corner exit speeds |
| Kart and driver weight | Not used | Grip, braking and acceleration stand in |
| Edge margin | 0.9 m from each edge | How wide the line can run |
| Direction | Start/finish on the long straight, heading south | Numbering, entry and exit |
| Line goal | Smoothest path, not fastest lap time | Ignores kart-specific behaviour |

Speeds compare corners with each other. They are not lap targets.

## Layout

- `index.html`: the interactive page
- `data/track_geometry.json`: centerline, half-width, racing line points and speeds, corner positions (pixel coordinates of the cropped satellite image)
- `data/corners.json`: per-corner entry, apex, exit, radius and braking data
- `inputs/`: the satellite image and the AI-generated etched outline used as a style reference (it does not line up geometrically with the satellite, so it was not used for tracing)
- `pipeline/`: the scripts that produced everything

## Pipeline

Paths in the scripts point at the original working folder, so edit the input and output paths before rerunning. Order:

1. `seg6.py`: asphalt mask from local luminance contrast
2. `skel.py`: keep the large component of the mask
3. `center2.py`: centerline from hand-placed waypoints, snapped to the ridge of the mask
4. `edges.py`: left and right track edges from brightness gradients
5. `line.py`: minimum-curvature racing line within the edges
6. `corners.py`, `final.py`: curvature, speed profile, corner detection and per-corner data
7. `build_data.py` then `page_tpl.html`: package the data and build the page
