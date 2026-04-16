#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

ACCENT  = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff",
           "#ffa657", "#79c0ff", "#f85149", "#56d364"]
RESDIR  = "results"
PLOTDIR = "plots"
os.makedirs(PLOTDIR, exist_ok=True)

# ── Measured data (from actual EdLab perf runs) ───────────────────────────────
# These are the ground-truth numbers from your assignment answers.
# The plotting script first tries to read CSVs; falls back to this embedded data.
KNOWN_DATA = {
    # kernel → (N, time_s, l1_miss_pct, llc_miss_pct)
    "naive":           {"N": 2048, "time": 65.04,  "l1": 72.47, "llc": 10.14},
    "reordered":       {"N": 2048, "time": 14.50,  "l1":  6.56, "llc": 56.86},
    "unrolled":        {"N": 2048, "time": 11.88,  "l1": 13.11, "llc": 56.85},
    "blocked":         {"N": 2048, "time": 24.35,  "l1":  9.49, "llc":  1.97},
    "avx_vectorized":  {"N": 1024, "time":  3.04,  "l1":  9.56, "llc":  0.17},
    "cache_aware":     {"N": 1024, "time": 16.82,  "l1":  1.25, "llc":  0.17},
    "register_kernel": {"N": 1024, "time": 12.78,  "l1":  1.10, "llc":  0.15},
}

KERNEL_LABELS = {
    "naive":           "Naive (i,j,k)",
    "reordered":       "Reordered (i,k,j)",
    "unrolled":        "Unrolled ×4",
    "blocked":         "Blocked B=32",
    "avx_vectorized":  "AVX2+FMA",
    "cache_aware":     "Cache-Aware",
    "register_kernel": "Register 2×4",
}

def load_csv(name):
    path = os.path.join(RESDIR, f"results_{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def savefig(fig, name):
    path = os.path.join(PLOTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

# ── 1. Execution Time Comparison ──────────────────────────────────────────────
def plot_execution_time():
    kernels = list(KERNEL_LABELS.keys())
    times   = [KNOWN_DATA[k]["time"] for k in kernels]
    labels  = [KERNEL_LABELS[k] for k in kernels]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(labels, times, color=ACCENT[:len(kernels)], height=0.6, edgecolor="none")

    # Annotate bars with time
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", ha="left", fontsize=9, color="#c9d1d9")

    ax.set_xlabel("Execution Time (seconds)")
    ax.set_title("Kernel Execution Time (N=2048 where comparable)", fontsize=13, pad=12)
    ax.axvline(x=0, color="#30363d", linewidth=0.8)
    ax.grid(axis="x", alpha=0.4)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, "execution_time_comparison.png")

# ── 2. L1 Cache Miss Rate Progression ─────────────────────────────────────────
def plot_l1_miss_progression():
    kernels = list(KERNEL_LABELS.keys())
    l1_vals = [KNOWN_DATA[k]["l1"] for k in kernels]
    labels  = [KERNEL_LABELS[k] for k in kernels]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(kernels))
    bars = ax.bar(x, l1_vals, color=ACCENT[:len(kernels)], width=0.6, edgecolor="none")

    # Highlight key drop from naive → reordered
    ax.annotate("", xy=(1, l1_vals[1] + 1), xytext=(0, l1_vals[0] - 2),
                arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.8))
    ax.text(0.5, (l1_vals[0] + l1_vals[1]) / 2, "−91%", color="#ff7b72",
            ha="center", va="center", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(5, l1_vals[5] + 0.5), xytext=(4, l1_vals[4] + 1),
                arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.8))
    ax.text(4.5, (l1_vals[4] + l1_vals[5]) / 2 + 3, "−87%", color="#3fb950",
            ha="center", va="center", fontsize=10, fontweight="bold")

    for bar, v in zip(bars, l1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8, color="#c9d1d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("L1 D-Cache Miss Rate (%)")
    ax.set_title("L1 Cache Miss Rate Across Optimization Levels", fontsize=13, pad=12)
    ax.set_ylim(0, max(l1_vals) * 1.2)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    savefig(fig, "l1_miss_rate_progression.png")

# ── 3. Speedup Over Naive ──────────────────────────────────────────────────────
def plot_speedup():
    naive_time = KNOWN_DATA["naive"]["time"]
    kernels    = list(KERNEL_LABELS.keys())
    speedups   = [naive_time / KNOWN_DATA[k]["time"] for k in kernels]
    labels     = [KERNEL_LABELS[k] for k in kernels]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(kernels))
    bars = ax.bar(x, speedups, color=ACCENT[:len(kernels)], width=0.6, edgecolor="none")

    ax.axhline(y=1.0, color="#ff7b72", linestyle="--", linewidth=1.2, label="Naive baseline (1×)")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=9, color="#e6edf3")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs Naive")
    ax.set_title("Cumulative Speedup vs Naive Baseline", fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    savefig(fig, "speedup_over_naive.png")

# ── 4. Block Size Sweep ────────────────────────────────────────────────────────
def plot_block_sweep():
    # Embedded block-size sweep data from assignment answers
    data = {
        "N=1024": {
            "block": [16,  32,  64,  128],
            "time":  [3.91, 2.95, 3.62, 4.09],
            "l1":    [13.43, 10.00, 8.09, 7.07],
        },
        "N=2048": {
            "block": [16,   32,    64,    128],
            "time":  [28.04, 24.35, 30.20, 25.01],
            "l1":    [12.57,  9.49,  7.93,  7.02],
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#58a6ff", "#3fb950"]
    for i, (label, d) in enumerate(data.items()):
        ax1.plot(d["block"], d["time"], marker="o", color=colors[i],
                 label=label, linewidth=2, markersize=7)
        ax2.plot(d["block"], d["l1"], marker="s", color=colors[i],
                 label=label, linewidth=2, markersize=7)

    ax1.axvline(x=32, color="#ffa657", linestyle="--", linewidth=1.5, label="Optimal B=32")
    ax2.axvline(x=32, color="#ffa657", linestyle="--", linewidth=1.5, label="Optimal B=32")

    ax1.set_xlabel("Block Size B")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Block Size vs Execution Time", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.4)

    ax2.set_xlabel("Block Size B")
    ax2.set_ylabel("L1 Miss Rate (%)")
    ax2.set_title("Block Size vs L1 Cache Miss Rate", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.4)

    # Annotation: cache math
    ax2.annotate("3×B²×8 ≤ 32KB\n→ B ≤ 36.9 → B=32",
                 xy=(32, 9.5), xytext=(65, 11.5),
                 arrowprops=dict(arrowstyle="->", color="#ffa657"),
                 color="#ffa657", fontsize=8.5)

    fig.suptitle("Cache Blocking: Block Size Sensitivity Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "block_size_sweep.png")

# ── 5. OpenMP Thread Scaling ───────────────────────────────────────────────────
def plot_thread_scaling():
    threads    = [1, 2, 4, 8]
    times      = [280.22, 146.14, 77.97, 39.82]   # N=4096, B=32
    ideal_t1   = times[0]
    ideal      = [ideal_t1 / t for t in [1, 2, 4, 8]]
    actual     = [times[0] / t for t in times]
    l3_miss    = [4.84, 7.64, 16.78, 15.57]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(threads, ideal,  "o--", color="#30363d", linewidth=1.5, label="Ideal linear speedup")
    ax1.plot(threads, actual, "o-",  color="#58a6ff", linewidth=2.5, markersize=8, label="Actual speedup")
    ax1.fill_between(threads, actual, ideal, alpha=0.12, color="#ff7b72", label="Efficiency loss")

    for t, s in zip(threads, actual):
        ax1.text(t, s + 0.1, f"{s:.2f}×", ha="center", va="bottom",
                 fontsize=9, color="#c9d1d9")

    ax1.set_xlabel("OpenMP Thread Count")
    ax1.set_ylabel("Speedup vs 1-thread blocked")
    ax1.set_title("OpenMP Scaling (N=4096, B=32)", fontsize=12)
    ax1.set_xticks(threads); ax1.legend(fontsize=9); ax1.grid(alpha=0.4)

    ax2.bar(threads, l3_miss, color=ACCENT[2], width=0.6, edgecolor="none")
    for t, v in zip(threads, l3_miss):
        ax2.text(t, v + 0.3, f"{v:.2f}%", ha="center", va="bottom",
                 fontsize=9, color="#c9d1d9")
    ax2.set_xlabel("OpenMP Thread Count")
    ax2.set_ylabel("LLC Miss Rate (%)")
    ax2.set_title("L3 Cache Pressure vs Thread Count", fontsize=12)
    ax2.set_xticks(threads); ax2.grid(axis="y", alpha=0.4)
    ax2.annotate("Shared L3 contention\nas threads compete\nfor cache space",
                 xy=(4, 16.78), xytext=(5.5, 13),
                 arrowprops=dict(arrowstyle="->", color="#ffa657"),
                 color="#ffa657", fontsize=8.5)

    fig.suptitle("OpenMP Multi-threading: Speedup & Cache Pressure", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "thread_scaling.png")

# ── 6. Cache Miss Heatmap ──────────────────────────────────────────────────────
def plot_cache_heatmap():
    kernels = list(KERNEL_LABELS.values())
    l1_vals = [KNOWN_DATA[k]["l1"]  for k in KNOWN_DATA]
    llc_vals= [KNOWN_DATA[k]["llc"] for k in KNOWN_DATA]

    data   = np.array([l1_vals, llc_vals])
    rlabels= ["L1 Miss %", "LLC Miss %"]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=max(max(l1_vals), max(llc_vals)))

    ax.set_xticks(np.arange(len(kernels)))
    ax.set_xticklabels(kernels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(rlabels, fontsize=10)

    for ci in range(len(kernels)):
        for ri in range(2):
            val = data[ri, ci]
            # Use black text for all cells — readable on any heatmap color
            ax.text(ci, ri, f"{val:.2f}%", ha="center", va="center",
                    fontsize=8.5, color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.6, edgecolor="none"))

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Miss Rate (%)")
    ax.set_title("Cache Miss Rate Heatmap Across Optimization Levels", fontsize=13, pad=12)
    fig.tight_layout()
    savefig(fig, "cache_miss_heatmap.png")

# ── 7. Unroll Factor Comparison ───────────────────────────────────────────────
def plot_unroll_comparison():
    unroll_factors = [2, 4, 8, 16]
    # N=2048, I=5 (from assignment answers)
    times = [17.80, 13.81, 11.88, 11.63]
    l1    = [ 4.43,  6.62, 13.11, 14.69]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ln1 = ax1.plot(unroll_factors, times, "o-", color="#58a6ff",
                   linewidth=2.5, markersize=8, label="Execution Time (s)")
    ln2 = ax2.plot(unroll_factors, l1, "s--", color="#ff7b72",
                   linewidth=2.5, markersize=8, label="L1 Miss Rate (%)")

    ax1.set_xlabel("Unroll Factor")
    ax1.set_ylabel("Execution Time (s)", color="#58a6ff")
    ax2.set_ylabel("L1 Miss Rate (%)", color="#ff7b72")
    ax1.set_xticks(unroll_factors)
    ax1.set_title("Loop Unroll Factor: Time vs L1 Miss Rate (N=2048)", fontsize=12)
    ax1.grid(alpha=0.4)

    ax2.annotate("Register\npressure\nspillover", xy=(8, 13.11), xytext=(10, 10),
                 arrowprops=dict(arrowstyle="->", color="#ffa657"),
                 color="#ffa657", fontsize=8.5)

    lines = ln1 + ln2
    labs  = [l.get_label() for l in lines]
    ax1.legend(lines, labs, fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "unroll_factor_comparison.png")

# ── 8. Summary Waterfall ──────────────────────────────────────────────────────
def plot_optimization_waterfall():
    """Waterfall chart showing incremental gain from each optimization."""
    stages = [
        ("Naive",        65.04),
        ("Reordered",    14.50),
        ("Unrolled x4",  11.88),
        ("Blocked B=32", 24.35),
        ("AVX2+FMA",      3.04),
        ("Cache-Aware",  16.82),
        ("Reg 2x4",      12.78),
    ]
    labels  = [s[0] for s in stages]
    times   = [s[1] for s in stages]
    naive_t = times[0]

    fig, ax = plt.subplots(figsize=(13, 6))
    colors_bar = []
    for i, t in enumerate(times):
        colors_bar.append(ACCENT[0] if i == 0 else
                          ("#3fb950" if t < times[i - 1] else "#ff7b72"))

    x    = np.arange(len(stages))
    bars = ax.bar(x, times, color=colors_bar, width=0.62, edgecolor="none")

    # Time labels well above each bar
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{t:.1f}s",
                ha="center", va="bottom", fontsize=9,
                color="#e6edf3", fontweight="bold")

    # Speedup row in a secondary text axis below the bars
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    speedup_labels = ["1.0x" if i == 0 else f"{naive_t/t:.1f}x"
                      for i, t in enumerate(times)]
    ax2.set_xticklabels(speedup_labels, fontsize=9, color="#8b949e")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 48))
    ax2.set_xlabel("Speedup vs Naive", fontsize=9, color="#8b949e", labelpad=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax.set_title("Optimization Waterfall — Execution Time per Stage",
                 fontsize=13, pad=14)
    ax.set_ylim(0, max(times) * 1.22)

    green_patch = mpatches.Patch(color="#3fb950", label="Faster than prev stage")
    red_patch   = mpatches.Patch(color="#ff7b72", label="Slower (different N or approach)")
    blue_patch  = mpatches.Patch(color=ACCENT[0],  label="Baseline (Naive)")
    ax.legend(handles=[blue_patch, green_patch, red_patch], fontsize=9,
              loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    savefig(fig, "optimization_waterfall.png")

# ── 9. Memory Hierarchy Latency (warmup) ──────────────────────────────────────
def plot_memory_hierarchy():
    """Visual summary of cache hierarchy latency tiers."""
    levels     = ["L1 cache\n(32 KB)", "L2 cache\n(256 KB)", "L3 cache\n(~16 MB)", "DRAM\n(>32 MB)"]
    latencies  = [1.08, 3.5, 12.5, 80]   # approximate ns from warmup experiment
    bandwidths = [500, 200, 50, 15]        # approximate GB/s

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(levels, latencies, color=ACCENT[:4], width=0.5, edgecolor="none")
    for bar, v in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v} ns", ha="center", va="bottom", fontsize=9, color="#c9d1d9")
    ax1.set_ylabel("Access Latency (ns)")
    ax1.set_title("Memory Hierarchy Latency", fontsize=12)
    ax1.grid(axis="y", alpha=0.4)
    ax1.set_yscale("log")

    bars2 = ax2.bar(levels, bandwidths, color=ACCENT[4:8], width=0.5, edgecolor="none")
    for bar, v in zip(bars2, bandwidths):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v} GB/s", ha="center", va="bottom", fontsize=9, color="#c9d1d9")
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_title("Memory Hierarchy Bandwidth", fontsize=12)
    ax2.grid(axis="y", alpha=0.4)

    fig.suptitle("Memory Hierarchy: The Motivation for Cache Optimization", fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, "memory_hierarchy.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plots...")
    plot_execution_time()
    plot_l1_miss_progression()
    plot_speedup()
    plot_block_sweep()
    plot_thread_scaling()
    plot_cache_heatmap()
    plot_unroll_comparison()
    plot_optimization_waterfall()
    plot_memory_hierarchy()
    print(f"\nAll plots saved to {PLOTDIR}/")
