"""
main.py
-------
Entry point. Ties all modules together.

Run:
  python main.py --video path/to/video.mov

Flow:
  1. Parse config (config.py)
  2. Open video, auto-detect ball radius for calibration (detection.py)
  3. Loop over frames: detect balls → update tracker (detection.py + tracking.py)
  4. Build physics DataFrame (physics.py)
  5. Save CSV and all plots (plotting.py)
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from temp.config import parse_args
from detection import detect_balls, estimate_radius_from_frames
from physics import compute_scale, build_dataframe, total_energy
from plotting import (
    plot_positions, plot_velocities, plot_energies,
    plot_energy_decay, save_csv, save_summary,
)
from temp.tracking import BallTracker


# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()

    # ── 1. Open video ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {cfg.video}")

    fps_raw      = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # After skip_frames, effective fps is lower
    fps_eff = fps_raw / cfg.skip_frames

    print(f"\n🎱 Newton's Cradle Tracker")
    print(f"   Video : {width}×{height} @ {fps_raw:.1f} fps  ({total_frames} frames)")
    print(f"   Duration : {total_frames / fps_raw:.1f} s")
    print(f"   Processing every {cfg.skip_frames} frame(s)  → effective {fps_eff:.1f} fps\n")

    # ── 2. Calibration — find ball radius in pixels ───────────────────────────
    # Rough estimate: assume 5 balls together span ~40% of frame width
    rough_r = int(width * 0.04)

    calib_circles = estimate_radius_from_frames(cap, rough_r, n_frames=30)

    if cfg.calib_diameter_px > 0:
        # User provided explicit pixel size → most accurate
        expected_r = cfg.calib_diameter_px // 2
    elif calib_circles:
        expected_r = int(np.median([c[2] for c in calib_circles]))
    else:
        expected_r = rough_r
        warnings.warn("Auto-calibration failed — using rough radius estimate. "
                      "Try --calib_diameter_px if results look wrong.")

    scale = compute_scale(
        calib_circles if calib_circles else [(0, 0, expected_r)],
        cfg.ball_diameter_m,
    )

    print(f"   Ball radius  : {expected_r} px")
    print(f"   Scale        : {scale * 1000:.3f} mm/px  ({1/scale:.0f} px/m)\n")

    # ── 3. Main processing loop ───────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind to start

    tracker  = BallTracker(n_balls=5)
    frame_id = 0
    max_dist = expected_r * 2.5           # tracking gate

    pbar = tqdm(total=total_frames // cfg.skip_frames,
                desc="Tracking", unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % cfg.skip_frames == 0:
            circles = detect_balls(frame, expected_r)
            tracker.update(frame_id, circles, max_dist_px=max_dist)

            # Optional live preview
            if cfg.preview:
                vis = frame.copy()
                for (cx, cy, r) in circles:
                    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
                cv2.imshow("Tracker  [q to quit]",
                           cv2.resize(vis, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

        frame_id += 1

    pbar.close()
    cap.release()
    if cfg.preview:
        cv2.destroyAllWindows()

    # ── 4. Physics ────────────────────────────────────────────────────────────
    print("\n[INFO] Computing physics…")
    raw = tracker.get_raw()
    if not raw:
        sys.exit("[ERROR] No tracks built. Check detection — try --preview to debug.")

    df    = build_dataframe(raw, fps_eff, scale, cfg.ball_mass)
    total = total_energy(df)

    # ── 5. Outputs ────────────────────────────────────────────────────────────
    print("[INFO] Saving outputs…")
    save_csv(df, cfg.output_dir)
    plot_positions(df, cfg.output_dir)
    plot_velocities(df, cfg.output_dir)
    plot_energies(df, total, cfg.output_dir)
    plot_energy_decay(total, cfg.output_dir)
    save_summary(df, total, fps_eff, scale,
                 cfg.ball_mass, cfg.ball_diameter_m,
                 cfg.string_length, cfg.output_dir)

    print(f"\n✅ Done! Results saved to:  {Path(cfg.output_dir).resolve()}/\n")


if __name__ == "__main__":
    main()
