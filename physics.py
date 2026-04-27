"""
physics.py
----------
Converts raw pixel tracking data into physical quantities:
  - pixel → metre calibration
  - height above equilibrium
  - velocity (via Savitzky-Golay smoothed derivative)
  - kinetic energy, potential energy, total energy

Output: a pandas DataFrame with one row per (frame, ball).

Column reference:
  frame   — frame number in video
  ball    — ball index 0-4 (left to right)
  time    — seconds from start
  x_px    — horizontal pixel position
  y_px    — vertical pixel position (image coords: 0 = top)
  x_m     — horizontal position [metres]
  h_m     — height above rest position [metres]  (positive = displaced up)
  vx      — horizontal velocity [m/s]
  vy      — vertical velocity [m/s]
  speed   — |v| = sqrt(vx²+vy²) [m/s]
  KE      — kinetic energy [J]
  PE      — potential energy [J]
  E       — total mechanical energy = KE + PE [J]
  mass    — ball mass [kg]
"""

import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from temp.config import G, SG_WINDOW, SG_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale(detected_circles: list[tuple[int, int, int]],
                  ball_diameter_m: float) -> float:
    """
    Returns metres-per-pixel.

    We know the real diameter of one ball (measured with ruler).
    We measure the detected radius in pixels.
    → scale = (real_radius_m) / (detected_radius_px)

    Uses the median radius across all detected balls (more robust than single ball).
    """
    if not detected_circles:
        raise ValueError("No circles provided for calibration.")
    median_r_px = float(np.median([c[2] for c in detected_circles]))
    real_r_m    = ball_diameter_m / 2.0
    return real_r_m / median_r_px   # [m/px]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(raw_tracks: dict,
                    fps: float,
                    scale: float,
                    ball_mass: float) -> pd.DataFrame:
    """
    Takes raw tracking output from BallTracker.get_raw() and returns
    a fully computed physics DataFrame.

    raw_tracks: {ball_id: {"frames": [...], "positions": [(x,y), ...]}}
    fps:        frames per second of the video (after skip_frames applied)
    scale:      metres per pixel
    ball_mass:  kg per ball
    """
    # ── 1. Flatten into rows ──────────────────────────────────────────────────
    rows = []
    for ball_id, data in raw_tracks.items():
        for frame, (xp, yp) in zip(data["frames"], data["positions"]):
            rows.append({"frame": frame, "ball": ball_id,
                         "x_px": xp, "y_px": yp})

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .sort_values(["frame", "ball"])
            .reset_index(drop=True))

    df["time"] = df["frame"] / fps
    df["mass"] = ball_mass

    # ── 2. Pixel → metre conversion ───────────────────────────────────────────
    # x_m: horizontal position (positive = right)
    df["x_m"] = df["x_px"] * scale

    # h_m: HEIGHT above equilibrium (positive = ball is raised)
    # In image coordinates y=0 is the TOP of the frame, so a raised ball
    # has a SMALLER y_px than at rest.
    # Equilibrium y_px = median y-position of each ball over the whole video
    # (more robust than first-frame only).
    eq_y = {}
    for b in df["ball"].unique():
        # Use the 10th percentile — the lowest point the ball reaches = rest pos
        eq_y[b] = float(np.percentile(df.loc[df["ball"] == b, "y_px"], 10))

    df["h_m"] = df.apply(
        lambda r: (eq_y[r["ball"]] - r["y_px"]) * scale,
        axis=1
    )

    # ── 3. Velocity via Savitzky-Golay derivative ─────────────────────────────
    # Raw positions are noisy (±1 px jitter from detection).
    # Simple finite difference (x[i+1]-x[i])/dt amplifies this noise badly.
    # Savitzky-Golay fits a polynomial to a sliding window and differentiates
    # the polynomial — gives smooth, physically meaningful velocity.
    vx_arr = np.zeros(len(df))
    vy_arr = np.zeros(len(df))

    for b in df["ball"].unique():
        idx = df.index[df["ball"] == b].tolist()
        if len(idx) < 7:
            continue

        t_arr = df.loc[idx, "time"].values
        x_arr = df.loc[idx, "x_m"].values
        h_arr = df.loc[idx, "h_m"].values

        dt   = float(np.mean(np.diff(t_arr)))
        win  = _sg_window(len(idx))

        try:
            vx_arr[idx] = savgol_filter(x_arr, win, SG_ORDER, deriv=1, delta=dt)
            vy_arr[idx] = savgol_filter(h_arr, win, SG_ORDER, deriv=1, delta=dt)
        except Exception:
            # Fallback to simple numpy gradient if SG fails
            warnings.warn(f"SG filter failed for ball {b}, using numpy.gradient")
            vx_arr[idx] = np.gradient(x_arr, t_arr)
            vy_arr[idx] = np.gradient(h_arr, t_arr)

    df["vx"]    = vx_arr
    df["vy"]    = vy_arr
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)

    # ── 4. Energies ───────────────────────────────────────────────────────────
    # KE = ½ m v²      [Joules]
    # PE = m g h       [Joules]  h clipped at 0 (no negative PE)
    # E  = KE + PE
    df["KE"] = 0.5 * ball_mass * df["speed"]**2
    df["PE"] = ball_mass * G * df["h_m"].clip(lower=0)
    df["E"]  = df["KE"] + df["PE"]

    return df


def total_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sums KE, PE, E across all 5 balls per frame.
    Returns a DataFrame with columns: frame, time, KE, PE, E
    """
    return (
        df.groupby("frame")[["KE", "PE", "E", "time"]]
          .agg({"KE": "sum", "PE": "sum", "E": "sum", "time": "first"})
          .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sg_window(n_points: int) -> int:
    """Return a valid Savitzky-Golay window length (odd, ≤ n_points, ≤ SG_WINDOW)."""
    win = min(SG_WINDOW, n_points)
    if win % 2 == 0:
        win -= 1
    return max(win, SG_ORDER + 2 if (SG_ORDER + 2) % 2 == 1 else SG_ORDER + 3)
