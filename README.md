# IPT-7 Double-Decker Newton's Cradle

Computer vision and physics analysis pipeline for a 5-ball Newton's cradle experiment.

The code detects ball positions in video, computes velocities and mechanical energies, and generates plots/CSV output for further analysis.

This repository is designed as a small analysis library: the files expose reusable building blocks for computer vision, calibration, kinematics, and plotting that can be reused across multiple experiments.

## Project goals

1. Track 5 silver balls through time in a high-speed video.
2. Convert pixel motion to physical units (meters, m/s, Joules).
3. Compute per-ball and total energy over time.
4. Estimate damping by fitting an exponential energy decay model.

## Physical situation

Experiment setup described in design notes:

- Five steel/silver balls in a Newton's cradle.
- Rightmost ball is released from about 45 degrees.
- Video recorded on phone (MOV, often 4K/60 FPS).
- Background is cardboard or matte surface.

Main physics quantities:

- Position: x(t), h(t)
- Velocity: v_x(t), v_y(t), |v|(t)
- Kinetic energy: KE = 0.5*m*v^2
- Potential energy: PE = m*g*h
- Mechanical energy: E = KE + PE

Total energy is summed across all five balls for each frame.

## Why OpenCV (and not DLC/TrackPy)

The chosen approach is classical OpenCV because the visual contrast is strong (silver balls vs darker background):

- No model training required.
- Fast and transparent pipeline.
- Good robustness with Hough circles + contour fallback.

DeepLabCut was considered unnecessary for this contrast level; TrackPy is less suited to this rigid 5-object geometry.

## Code structure

- `main.py`
	- Orchestrates the full pipeline.
	- Opens video, calibrates scale, runs detection/tracking loop.
	- Builds physics table and saves all outputs.

- `detection.py`
	- HSV masking for bright low-saturation "silver" pixels.
	- Primary detector: Hough circles.
	- Fallback detector: contour circularity filtering.
	- Returns up to 5 balls sorted left to right.

- `physics.py`
	- Pixel-to-meter calibration from detected radius and known ball diameter.
	- Computes h(t), smoothed derivatives (Savitzky-Golay), KE/PE/E.
	- Aggregates total energy across all balls.

- `plotting.py`
	- Saves figures and summary artifacts.
	- Includes exponential fit: E(t) = E0*exp(-gamma*t).

## Library-style function reference

### main.py

- `main()`
	- High-level orchestrator for the full workflow.
	- Reads config, runs calibration and frame loop, computes physics tables, and saves artifacts.

### detection.py

- `build_mask(frame_bgr)`
	- Produces a binary mask for likely silver-ball pixels in HSV space.
- `detect_balls(frame_bgr, expected_r)`
	- Main detector returning up to 5 circles sorted from left to right.
- `estimate_radius_from_frames(cap, rough_r, n_frames=30)`
	- Quick calibration helper that searches early frames for stable detections.

Internal helpers used by `detect_balls`:

- `_hough_circles(frame_bgr, mask, expected_r)`
	- Primary precise circle detector.
- `_contour_circles(mask, expected_r)`
	- Robust fallback based on contour area and circularity.

### physics.py

- `compute_scale(detected_circles, ball_diameter_m)`
	- Converts pixel units to meters using known ball diameter.
- `build_dataframe(raw_tracks, fps, scale, ball_mass)`
	- Core transformation from tracked positions to physical state variables.
	- Computes time, metric coordinates, velocities, and energies.
- `total_energy(df)`
	- Aggregates system-level KE/PE/E per frame.
- `_sg_window(n_points)`
	- Utility to generate a valid Savitzky-Golay window size.

### plotting.py

- `plot_positions(df, output_dir)`
	- Plots x(t) and h(t) for all balls.
- `plot_velocities(df, output_dir)`
	- Plots speed magnitude over time.
- `plot_energies(df, total, output_dir)`
	- Plots per-ball energy and total KE/PE/E.
- `plot_energy_decay(total, output_dir)`
	- Fits and visualizes exponential damping of total mechanical energy.
- `save_csv(df, output_dir)`
	- Exports frame-level processed data for downstream analysis.
- `save_summary(df, total, fps, scale, ball_mass, ball_diameter_m, string_length, output_dir)`
	- Writes human-readable experiment summary metrics.
- `_save(fig, output_dir, filename)`
	- Shared output helper for consistent figure saving.

## Reuse philosophy

The recommended way to use this project as a library is:

1. Reuse `detection.py` for object extraction from new experiment videos.
2. Reuse `physics.py` for unit conversion, derivatives, and energy computation.
3. Reuse `plotting.py` for consistent reporting artifacts.
4. Keep `main.py` as an example pipeline entry point that can be adapted per experiment.

## Processing pipeline

1. Read video and metadata (FPS, width, height).
2. Auto-estimate ball radius from early frames (or use manual calibration diameter).
3. Detect balls per processed frame.
4. Track identities frame-to-frame.
5. Convert pixel trajectories to metric trajectories.
6. Differentiate smoothed trajectories to get velocity.
7. Compute KE, PE, E per ball and total energy.
8. Save plots, CSV, and text summary.

## Outputs

Typical output files:

- `ball_tracking.csv` - frame-level data table.
- `positions.png` - x(t), h(t) for all balls.
- `velocities.png` - speed(t) for all balls.
- `energies.png` - per-ball and total energy.
- `energy_decay.png` - exponential fit to total energy envelope.
- `summary.txt` - scalar metrics (max velocity, max energy, retention).

## Measurement assumptions and limitations

1. The algorithm assumes approximately frontal camera view.
2. Perspective error increases if camera is too close.
3. With cradle half-width ~18 cm and camera distance ~35 cm, edge-ball compression can be around 11 percent.
4. Better geometry: move camera back (about 120-150 cm) and use optical zoom.
5. High FPS is more important than very high resolution for velocity accuracy.

## Filming recommendations

- Keep camera level relative to cradle bar.
- Use diffuse lighting to avoid strong reflections.
- Matte black background behind balls improves segmentation.
- Prefer 60 FPS minimum; 240 FPS slow motion is excellent for early fast collisions.
- Record 1-2 seconds of rest state before release if equilibrium referencing is needed.

## Run notes

Current files import modules from a `temp` package (`temp.config`, `temp.tracking`).
Make sure those modules exist in your working project tree before running `main.py`.

## Repository

GitHub:

https://github.com/Danyil-Dorosh/IPT-7-Double-Decker-Newton-s-Cradle.git
