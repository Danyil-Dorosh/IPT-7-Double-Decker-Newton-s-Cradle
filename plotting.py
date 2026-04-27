"""
plotting.py
-----------
All matplotlib figures. Each function saves one PNG to output_dir.

Functions:
  plot_positions(df, output_dir)   → positions.png
  plot_velocities(df, output_dir)  → velocities.png
  plot_energies(df, total, output_dir) → energies.png
  plot_energy_decay(total, output_dir) → energy_decay.png  (+ exponential fit)
  save_csv(df, output_dir)         → ball_tracking.csv
  save_summary(df, total, ...)     → summary.txt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed; remove this line if you want interactive plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from temp.config import BALL_COLORS, BALL_LABELS
from physics import _sg_window


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_positions(df: pd.DataFrame, output_dir: str):
    """
    Two panels:
      Top:    x(t) — horizontal position of each ball [cm]
      Bottom: h(t) — height above rest position [cm]
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for b in range(5):
        sub = df[df["ball"] == b]
        if sub.empty:
            continue
        axes[0].plot(sub["time"], sub["x_m"] * 100,
                     color=BALL_COLORS[b], lw=1.2, alpha=0.85, label=BALL_LABELS[b])
        axes[1].plot(sub["time"], sub["h_m"] * 100,
                     color=BALL_COLORS[b], lw=1.2, alpha=0.85)

    axes[0].set_ylabel("Horizontal position x (cm)")
    axes[1].set_ylabel("Height above rest h (cm)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title("Ball Positions over Time", fontweight="bold")
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir, "positions.png")


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocities(df: pd.DataFrame, output_dir: str):
    """Speed |v| in m/s for each ball."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for b in range(5):
        sub = df[df["ball"] == b]
        if sub.empty:
            continue
        ax.plot(sub["time"], sub["speed"],
                color=BALL_COLORS[b], lw=1.2, alpha=0.85, label=BALL_LABELS[b])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Ball Speeds over Time", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "velocities.png")


# ─────────────────────────────────────────────────────────────────────────────
# ENERGIES  (key physics plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_energies(df: pd.DataFrame, total: pd.DataFrame, output_dir: str):
    """
    Two panels:
      Top:    mechanical energy per ball [mJ]
      Bottom: TOTAL KE + PE + E across all balls [mJ]  ← the main physics plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Per-ball energy
    for b in range(5):
        sub = df[df["ball"] == b]
        if sub.empty:
            continue
        axes[0].plot(sub["time"], sub["E"] * 1000,
                     color=BALL_COLORS[b], lw=1.5, alpha=0.8, label=BALL_LABELS[b])

    axes[0].set_ylabel("Energy per ball (mJ)")
    axes[0].set_title("Per-Ball Mechanical Energy  (KE + PE)", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Total energy with KE/PE fill
    t = total["time"]
    axes[1].fill_between(t, total["KE"] * 1000, alpha=0.35, color="#457B9D", label="Total KE")
    axes[1].fill_between(t, total["PE"] * 1000, alpha=0.35, color="#E9C46A", label="Total PE")
    axes[1].plot(t, total["E"] * 1000,
                 color="#E63946", lw=2.2, label="Total E = KE + PE", zorder=5)

    # Smoothed trend line
    E_vals = total["E"].values * 1000
    win    = _sg_window(len(E_vals))
    if win >= 5:
        E_smooth = savgol_filter(E_vals, win, 3)
        axes[1].plot(t, E_smooth, color="black", lw=1.5, ls="--",
                     alpha=0.6, label="E (smoothed)")

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Total mechanical energy (mJ)")
    axes[1].set_title("Total Mechanical Energy over Time", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir, "energies.png")


# ─────────────────────────────────────────────────────────────────────────────
# ENERGY DECAY FIT
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_decay(total: pd.DataFrame, output_dir: str):
    """
    Fits E(t) = E₀ · exp(−γ·t) to the total energy curve.
    γ is the damping coefficient [1/s].
    τ = 1/γ is the energy half-life [s].
    """
    def exp_decay(t, E0, gamma):
        return E0 * np.exp(-gamma * t)

    t = total["time"].values
    E = total["E"].values * 1000   # mJ

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, E, color="#457B9D", lw=1.5, alpha=0.7, label="Total E (measured)")

    try:
        popt, _ = curve_fit(exp_decay, t, E,
                            p0=[E.max(), 0.01],
                            bounds=([0, 0], [np.inf, 10]))
        E0_fit, gamma_fit = popt
        tau = 1.0 / gamma_fit
        print(f"[FIT] E₀ = {E0_fit:.2f} mJ  |  γ = {gamma_fit:.4f} s⁻¹  |  τ = {tau:.1f} s")

        t_fine = np.linspace(t.min(), t.max(), 2000)
        label  = rf"Fit: $E_0 e^{{-\gamma t}}$,  γ={gamma_fit:.4f} s⁻¹,  τ={tau:.1f} s"
        ax.plot(t_fine, exp_decay(t_fine, E0_fit, gamma_fit),
                color="#E63946", lw=2, ls="--", label=label)
    except Exception as e:
        print(f"[WARN] Exponential fit failed: {e}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total mechanical energy (mJ)")
    ax.set_title("Energy Dissipation — Exponential Decay Fit", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "energy_decay.png")


# ─────────────────────────────────────────────────────────────────────────────
# CSV + SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, output_dir: str):
    path = Path(output_dir) / "ball_tracking.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[SAVED] {path}")


def save_summary(df: pd.DataFrame, total: pd.DataFrame,
                 fps: float, scale: float,
                 ball_mass: float, ball_diameter_m: float,
                 string_length: float, output_dir: str):
    e0   = total["E"].iloc[:10].mean() * 1000
    efin = total["E"].iloc[-10:].mean() * 1000

    lines = [
        "=" * 60,
        "NEWTON'S CRADLE — TRACKING ANALYSIS REPORT",
        "=" * 60,
        "",
        f"  Video FPS          : {fps:.2f}",
        f"  Scale              : {scale * 1000:.4f} mm/px",
        f"  Ball diameter      : {ball_diameter_m * 1000:.1f} mm",
        f"  String length      : {string_length * 100:.1f} cm",
        f"  Ball mass          : {ball_mass * 1000:.1f} g",
        f"  Duration tracked   : {df['time'].max():.2f} s",
        "",
        "-" * 60,
        "PER-BALL STATISTICS",
        "-" * 60,
    ]
    for b in range(5):
        sub = df[df["ball"] == b]
        if sub.empty:
            continue
        lines.append(
            f"  Ball {b+1}:  v_max={sub['speed'].max():.3f} m/s  "
            f"E_max={sub['E'].max()*1000:.2f} mJ  "
            f"h_max={sub['h_m'].max()*100:.1f} cm"
        )

    lines += [
        "",
        "-" * 60,
        "TOTAL ENERGY",
        "-" * 60,
        f"  Initial  E₀  : {e0:.2f} mJ",
        f"  Final    E_f : {efin:.2f} mJ",
        f"  Retained     : {100 * efin / e0:.1f} %" if e0 > 0 else "  (no data)",
        "",
        "=" * 60,
    ]

    path = Path(output_dir) / "summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, output_dir: str, filename: str):
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")
