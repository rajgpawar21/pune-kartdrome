# Pune Kartdrome: sketch and racing line

An etched-style vector sketch of Pune Kartdrome traced from a satellite image, with a minimum-curvature racing line through all 13 corners.

Link - https://rajgpawar21.github.io/pune-kartdrome/

Open `index.html` in a browser (it is self-contained). Switch the direction of travel between clockwise and counterclockwise, toggle Sketch/Satellite, racing line, centerline and corner numbers, and pick a corner to see turn-in, apex, exit and braking. To host it, enable GitHub Pages on this repo (Settings, Pages, deploy from branch, root).

## Results

- 12 corners in either direction: 7 right and 5 left clockwise, 5 right and 7 left counterclockwise
- Lap length about 645 m (derived from the assumed width, not measured); model lap about 47.4 s clockwise and 47.5 s counterclockwise
- Corners are numbered in driving order from the start/finish on the long straight
- Switching direction re-solves the racing line, speed profile, braking zones, corner numbering and entry/exit. The line differs between directions by up to about 1.5 m. The centerline and track edges are physical, so they do not change

## Assumptions

The image has no scale and nothing is known about the kart, so these are estimates. Replace them with real figures to rescale everything.

| Item | Value used | Affects |
|---|---|---|
| Track width | 7 m (about 59 px, 0.118 m per pixel) | Every distance and radius |
| Lap length | about 645 m, derived | Not measured. Differs by a few metres between directions because the line differs |
| Kart top speed | 61 km/h (17 m/s) | Straight speeds |
| Cornering grip | 1.1 g (11 m/s²) | Corner speeds |
| Braking | 0.9 g (9 m/s²) | Brake zones |
| Acceleration | 4 m/s² | Corner exit speeds |
| Kart and driver weight | Not used | Grip, braking and acceleration stand in |
| Edge margin | 0.9 m from each edge | How wide the line can run |
| Direction of travel | Clockwise or counterclockwise, start/finish on the long straight | Re-solves the line, speeds, braking zones, numbering and entry/exit |
| Line goal | Fastest model lap, starting from the smoothest path | Optimised against the simple model above, so it ignores kart-specific behaviour |
| Corner detection | Radius under 25 m with a clear curvature peak | Gentle bends the line barely slows for are not numbered |

Speeds and the model lap time compare corners and directions with each other. They are not lap targets.

## Layout

- `index.html`: the interactive page
- `data/analysis.json`: shared track geometry plus, for each direction (`cw`, `ccw`), the racing line points and speeds, the corner list with entry, apex, exit, radius and braking data, and the model lap time (pixel coordinates of the cropped satellite image)
- `inputs/`: the satellite image and the AI-generated etched outline used as a style reference (it does not line up geometrically with the satellite, so it was not used for tracing)
- `pipeline/`: the scripts that produced everything

## Pipeline

Paths in the scripts point at the original working folder, so edit the input and output paths before rerunning. Order:

1. `seg6.py`: asphalt mask from local luminance contrast
2. `skel.py`: keep the large component of the mask
3. `center2.py`: centerline from hand-placed waypoints, snapped to the ridge of the mask
4. `edges.py`: left and right track edges from brightness gradients
5. `line.py`: smoothest (minimum-curvature) racing line within the edges
6. `directions.py`: for each direction, refine that line to minimise a quasi-steady-state lap time, then compute curvature, speeds, corner detection and per-corner data. Writes `analysis.json`
7. `build_page.py`: crop the satellite image and embed it with `analysis.json` into `page_tpl.html`, producing `index.html`

To change an assumption such as kart top speed, grip, braking, acceleration or track width, edit the constants at the top of `directions.py` and rerun steps 6 and 7.
