#!/usr/bin/env python3
"""
Generate NeurIPS appendix figures (E3–E8).

- No seaborn, one chart per file, no explicit colors (matplotlib defaults only).
- By default, uses synthetic data to demonstrate the look & structure.
- You can replace the synthetic arrays with your real outputs:
  - E3: supply baseline/method heatmaps (time x space).
  - E4: supply metric time series.
  - E5: supply per-category violation rates for baseline & ours.
  - E6: supply trajectory arrays (x,y) for ablations.
  - E7: supply a list of diverse (x,y) trajectories.
  - E8: supply a precomputed energy grid or callable.

Usage:
  python generate_neurips_appendix_figs.py --out figs --pdf
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------- Utils ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, pdf: bool = False) -> None:
    plt.tight_layout()
    path = path.with_suffix(".png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if pdf:
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# ---------------------------- E3: Collision Heatmaps ----------------------------

def generate_e3_collision_heatmaps(
    out_dir: Path,
    baseline: Optional[np.ndarray] = None,
    method: Optional[np.ndarray] = None,
    save_pdf: bool = False,
) -> Tuple[Path, Path]:
    """
    E3a/E3b: Collision heatmaps over time (rows) and space (cols).

    Args:
        baseline: (T, X) array (higher = worse). If None, uses synthetic.
        method:   (T, X) array. If None, uses synthetic.
    Returns:
        Paths to E3a and E3b PNGs (and PDFs if save_pdf=True).
    """
    ensure_dir(out_dir)

    if baseline is None or method is None:
        # Synthetic data (T x X grid)
        T, X = 80, 80
        x = np.linspace(-3, 3, X)
        t = np.linspace(0, 1, T)
        Xg, Tg = np.meshgrid(x, t)

        baseline = (
            1.4 * np.exp(-((Xg - 0.5*np.sin(4*Tg*np.pi))**2) / 0.8)
            + 1.2 * np.exp(-((Xg + 1.0*np.cos(3*Tg*np.pi))**2) / 0.5)
            + 0.6 * (0.3 + 0.7*Tg)
        )
        method = (
            0.9 * np.exp(-((Xg - 0.25*np.sin(3*Tg*np.pi))**2) / 1.1)
            + 0.6 * np.exp(-((Xg + 0.8*np.cos(2*Tg*np.pi))**2) / 0.9)
            + 0.25 * (0.3 + 0.2*Tg)
        )

    # Normalize for fair visual comparison
    def norm(a):
        a = a.astype(float)
        return (a - a.min()) / (a.max() - a.min() + 1e-12)

    baseline_n = norm(baseline)
    method_n = norm(method)

    # E3a: Baseline
    plt.figure()
    plt.imshow(baseline_n, aspect='auto', origin='lower')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Collision Potential (Baseline)")
    p3a = out_dir / "Figure_E3a_baseline_heatmap"
    savefig(p3a, pdf=save_pdf)

    # E3b: Ours
    plt.figure()
    plt.imshow(method_n, aspect='auto', origin='lower')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Collision Potential (Ours)")
    p3b = out_dir / "Figure_E3b_method_heatmap"
    savefig(p3b, pdf=save_pdf)

    return p3a.with_suffix(".png"), p3b.with_suffix(".png")


# ---------------------------- E4: Temporal Consistency ----------------------------

@dataclass
class TemporalMetrics:
    t_sec: np.ndarray
    baseline_collision: np.ndarray
    ours_collision: np.ndarray
    baseline_kin_valid: np.ndarray
    ours_kin_valid: np.ndarray
    baseline_overall: np.ndarray
    ours_overall: np.ndarray


def generate_e4_temporal_consistency(
    out_dir: Path,
    metrics: Optional[TemporalMetrics] = None,
    save_pdf: bool = False,
) -> Path:
    """
    E4: Time series of collision rate (lower better), kinematic validity, overall validity.
    """
    ensure_dir(out_dir)

    if metrics is None:
        t_sec = np.linspace(0, 9, 181)
        baseline_collision = 0.10 + 0.20*(t_sec/9.0) + 0.02*np.sin(1.7*t_sec)
        ours_collision     = 0.06 + 0.03*(t_sec/9.0) + 0.01*np.sin(1.4*t_sec + 0.5)
        baseline_kin_valid = 0.95 - 0.12*(t_sec/9.0) - 0.01*np.cos(1.3*t_sec)
        ours_kin_valid     = 0.97 - 0.03*(t_sec/9.0) - 0.005*np.cos(1.2*t_sec + 0.3)
        baseline_overall   = 0.90 - 0.15*(t_sec/9.0) - 0.02*np.sin(0.9*t_sec + 0.1)
        ours_overall       = 0.95 - 0.05*(t_sec/9.0) - 0.01*np.sin(0.8*t_sec + 0.2)
        metrics = TemporalMetrics(
            t_sec, baseline_collision, ours_collision,
            baseline_kin_valid, ours_kin_valid,
            baseline_overall, ours_overall
        )

    plt.figure()
    plt.plot(metrics.t_sec, metrics.baseline_collision, label="Collision Rate (Baseline)")
    plt.plot(metrics.t_sec, metrics.ours_collision,     label="Collision Rate (Ours)")
    plt.plot(metrics.t_sec, metrics.baseline_kin_valid, label="Kinematic Validity (Baseline)")
    plt.plot(metrics.t_sec, metrics.ours_kin_valid,     label="Kinematic Validity (Ours)")
    plt.plot(metrics.t_sec, metrics.baseline_overall,   label="Overall Validity (Baseline)")
    plt.plot(metrics.t_sec, metrics.ours_overall,       label="Overall Validity (Ours)")
    plt.xlabel("Time (s)")
    plt.ylabel("Metric Value")
    plt.title("Temporal Consistency of Validity Metrics")
    plt.legend(ncol=2, fontsize=8)
    p4 = out_dir / "Figure_E4_temporal_consistency"
    savefig(p4, pdf=save_pdf)
    return p4.with_suffix(".png")


# ---------------------------- E5: Constraint Violation Breakdown ----------------------------

def generate_e5_constraint_breakdown(
    out_dir: Path,
    categories: Optional[List[str]] = None,
    baseline_vals: Optional[np.ndarray] = None,
    ours_vals: Optional[np.ndarray] = None,
    save_pdf: bool = False,
) -> Path:
    """
    E5: Grouped bar chart showing violation rates per category (lower better).
    """
    ensure_dir(out_dir)

    if categories is None:
        categories = ["Collisions", "Off-road", "Kinematic", "Speeding", "Red-light"]
    n = len(categories)

    if baseline_vals is None or ours_vals is None:
        baseline_vals = np.array([0.18, 0.14, 0.20, 0.10, 0.08])
        ours_vals     = np.array([0.07, 0.06, 0.08, 0.05, 0.03])

    assert len(baseline_vals) == n and len(ours_vals) == n, "Value arrays must match categories length."

    idx = np.arange(n)
    bar_width = 0.35

    plt.figure()
    plt.bar(idx - bar_width/2, baseline_vals, width=bar_width, label="Baseline")
    plt.bar(idx + bar_width/2, ours_vals,     width=bar_width, label="Ours")
    plt.xticks(idx, categories, rotation=20)
    plt.ylabel("Violation Rate")
    plt.title("Constraint Violation Breakdown")
    plt.legend()
    p5 = out_dir / "Figure_E5_constraint_breakdown"
    savefig(p5, pdf=save_pdf)
    return p5.with_suffix(".png")


# ---------------------------- E6: Ablation Visualization ----------------------------

def synthetic_trajectory(seed_shift: float = 0.0, smoothness: float = 0.1, collision_bias: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, 200)
    x = t * 30.0
    y = 3*np.sin(0.6*np.pi*t) + 2*np.sin(1.2*np.pi*t + seed_shift)
    y += collision_bias * np.exp(-((x-15.0)**2)/20.0)
    if smoothness < 1.0:
        k = max(1, int(1/smoothness))
        y = np.convolve(y, np.ones(k)/k, mode='same')
    return x, y


def generate_e6_ablation_panels(out_dir: Path, save_pdf: bool = False) -> List[Path]:
    """
    E6a–E6d: Four separate panels for visual ablations.
    """
    ensure_dir(out_dir)
    outputs = []

    # (a) Full model
    x, y = synthetic_trajectory(seed_shift=0.1, smoothness=0.2, collision_bias=-0.4)
    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("(a) Full Model")
    plt.legend()
    p6a = out_dir / "Figure_E6a_full_model"
    savefig(p6a, pdf=save_pdf)
    outputs.append(p6a.with_suffix(".png"))

    # (b) No collision potential
    x, y = synthetic_trajectory(seed_shift=0.5, smoothness=0.3, collision_bias=+0.6)
    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("(b) No Collision Potential")
    plt.legend()
    p6b = out_dir / "Figure_E6b_no_collision"
    savefig(p6b, pdf=save_pdf)
    outputs.append(p6b.with_suffix(".png"))

    # (c) No kinematic constraints
    t = np.linspace(0, 1, 200)
    x = t * 30.0
    y = 3*np.sin(0.6*np.pi*t) + 2*np.sin(1.0*np.pi*t + 0.2) + 0.8*np.sin(12*np.pi*t)
    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("(c) No Kinematic Constraints")
    plt.legend()
    p6c = out_dir / "Figure_E6c_no_kinematic"
    savefig(p6c, pdf=save_pdf)
    outputs.append(p6c.with_suffix(".png"))

    # (d) No graph attention
    x = t * 30.0
    y = 3*np.sin(0.6*np.pi*t) + 1.5*np.sin(2.2*np.pi*t) + 0.5*np.sin(8*np.pi*t) * (t > 0.35) * (t < 0.7)
    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("(d) No Graph Attention")
    plt.legend()
    p6d = out_dir / "Figure_E6d_no_graph"
    savefig(p6d, pdf=save_pdf)
    outputs.append(p6d.with_suffix(".png"))

    return outputs


# ---------------------------- E7: Diversity Visualization ----------------------------

def generate_e7_diversity(
    out_dir: Path,
    trajectories: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    save_pdf: bool = False,
) -> Path:
    """
    E7: Plot 5–10 diverse valid trajectories from the same initial state.
    """
    ensure_dir(out_dir)

    if trajectories is None:
        rng = np.random.default_rng(7)
        trajectories = []
        for _ in range(10):
            phase = rng.uniform(0, 2*np.pi)
            amp1  = 3 + 0.5*rng.normal()
            amp2  = 1.5 + 0.3*rng.normal()
            t = np.linspace(0, 1, 250)
            x = t * 40.0
            y = amp1*np.sin(0.6*np.pi*t + phase) + amp2*np.sin(1.1*np.pi*t + 0.3*phase)
            trajectories.append((x, y))

    plt.figure()
    for (x, y) in trajectories:
        plt.plot(x, y, lw=1.2)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Diversity of Valid Trajectories (Same Initial State)")
    p7 = out_dir / "Figure_E7_diversity"
    savefig(p7, pdf=save_pdf)
    return p7.with_suffix(".png")


# ---------------------------- E8: Energy Function Landscape ----------------------------

def generate_e8_energy_landscape(
    out_dir: Path,
    energy_grid: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    energy_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    save_pdf: bool = False,
) -> Path:
    """
    E8: Contour plot of trajectory energy landscape (lower = better).
    Provide either:
      - energy_grid with grid_x, grid_y, or
      - energy_fn callable(XX, YY) -> energy array
    If None, uses a synthetic valley with one repulsive obstacle.
    """
    ensure_dir(out_dir)

    if energy_grid is None:
        xx = np.linspace(-5, 25, 250)
        yy = np.linspace(-10, 10, 220)
        XX, YY = np.meshgrid(xx, yy)
        if energy_fn is None:
            def default_energy(XX, YY):
                valley = (YY**2) / (4 + 2*np.exp(-((XX-16.0)**2)/18.0)) + 0.4*np.sin(XX/2.5)*np.cos(YY/3.0)
                obstacle = 1.2*np.exp(-((XX-8.0)**2 + (YY-3.0)**2)/3.0)
                return valley + obstacle
            energy = default_energy(XX, YY)
        else:
            energy = energy_fn(XX, YY)
    else:
        assert grid_x is not None and grid_y is not None, "Provide grid_x, grid_y with energy_grid."
        XX, YY = np.meshgrid(grid_x, grid_y)
        energy = energy_grid

    plt.figure()
    plt.contourf(XX, YY, energy, levels=30)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Energy Function Landscape (Lower is Better)")
    p8 = out_dir / "Figure_E8_energy_landscape"
    savefig(p8, pdf=save_pdf)
    return p8.with_suffix(".png")


# ---------------------------- Main ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS appendix figures E3–E8.")
    parser.add_argument("--out", type=Path, default=Path("figs"), help="Output directory for figures.")
    parser.add_argument("--pdf", action="store_true", help="Also save PDF copies alongside PNGs.")
    args = parser.parse_args()

    ensure_dir(args.out)

    # E3
    p3a, p3b = generate_e3_collision_heatmaps(args.out, save_pdf=args.pdf)
    print("E3:", p3a, p3b)

    # E4
    p4 = generate_e4_temporal_consistency(args.out, save_pdf=args.pdf)
    print("E4:", p4)

    # E5
    p5 = generate_e5_constraint_breakdown(args.out, save_pdf=args.pdf)
    print("E5:", p5)

    # E6
    ablations = generate_e6_ablation_panels(args.out, save_pdf=args.pdf)
    print("E6:", *ablations)

    # E7
    p7 = generate_e7_diversity(args.out, save_pdf=args.pdf)
    print("E7:", p7)

    # E8
    p8 = generate_e8_energy_landscape(args.out, save_pdf=args.pdf)
    print("E8:", p8)


if __name__ == "__main__":
    main()
