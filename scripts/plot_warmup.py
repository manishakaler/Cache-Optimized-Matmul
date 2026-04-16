#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ACCENT = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#ffa657", "#79c0ff"]

RESDIR  = "results"
PLOTDIR = "plots"
os.makedirs(PLOTDIR, exist_ok=True)

def load_data():
    path = os.path.join(RESDIR, "cache_miss_data.csv")
    df = pd.read_csv(path)
    df["size_bytes"]     = df["array_size_in_elements"] * 8
    df["total_accesses"] = (df["array_size_in_elements"] / df["stride"]) * df["iterations"]
    df["ns_per_access"]  = (df["runtime_in_seconds"] * 1e9) / df["total_accesses"]
    df["l1_miss_rate"]   = df["l1_misses"]  / df["total_accesses"]
    df["l2_miss_rate"]   = df["l2_misses"]  / df["total_accesses"]
    df["l3_miss_rate"]   = df["l3_misses"]  / df["total_accesses"]
    return df

def savefig(fig, name):
    path = os.path.join(PLOTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

# ── 1. Stride vs Latency (512 MB array) ───────────────────────────────────────
def plot_stride_latency(df):
    # Largest array size in the dataset
    max_sz = df["array_size_in_elements"].max()
    sub = df[df["array_size_in_elements"] == max_sz].sort_values("stride")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub["stride"], sub["ns_per_access"],
            "o-", color=ACCENT[0], linewidth=2.5, markersize=7)

    # Annotate cache-line saturation point (stride=8)
    sat = sub[sub["stride"] == 8].iloc[0]
    ax.axvline(x=8, color=ACCENT[2], linestyle="--", linewidth=1.3,
               label="stride=8: every access\ncrosses a cache line (100% miss)")
    ax.annotate(f"{sat['ns_per_access']:.1f} ns",
                xy=(8, sat["ns_per_access"]),
                xytext=(18, sat["ns_per_access"] * 1.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT[2]),
                color=ACCENT[2], fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Stride (elements)")
    ax.set_ylabel("Latency (ns/access)")
    ax.set_title(f"Stride vs Access Latency  (array = {max_sz*8//1024//1024} MB)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    # Add stride labels on x-axis
    strides = sorted(sub["stride"].unique())
    ax.set_xticks(strides)
    ax.set_xticklabels([str(int(s)) for s in strides], fontsize=8)
    fig.tight_layout()
    savefig(fig, "stride_vs_latency.png")

# ── 2. L1/L2/L3 Miss Count vs Array Size (stride=1) ──────────────────────────
def plot_cache_levels(df):
    sub = df[df["stride"] == 1].sort_values("size_bytes")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, color in [
        ("l1_misses", "L1 misses", ACCENT[0]),
        ("l2_misses", "L2 misses", ACCENT[1]),
        ("l3_misses", "L3 misses", ACCENT[2]),
    ]:
        ax.plot(sub["size_bytes"] / 1024, sub[col],
                "o-", label=label, color=color, linewidth=2, markersize=6)

    # Mark L3 inflection: jump between 8MB and 32MB
    ax.axvspan(8*1024, 32*1024, alpha=0.08, color=ACCENT[2],
               label="L3 size range (~16–25 MB)")
    ax.axvline(x=25*1024, color=ACCENT[2], linestyle=":", linewidth=1,
               label="EdLab L3 spec: 25 MB")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("Total Cache Misses (count)")
    ax.set_title("Cache Misses vs Array Size  (stride=1, sequential-ish)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    savefig(fig, "cache_levels_miss_count.png")

# ── 3. L3 Miss Rate inflection → reveals L3 size ─────────────────────────────
def plot_l3_inflection(df):
    sub = df[df["stride"] == 1].sort_values("size_bytes").copy()
    sub = sub[sub["total_accesses"] > 0]
    sub["l3_pct"] = sub["l3_misses"] / sub["total_accesses"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub["size_bytes"] / 1024, sub["l3_pct"],
            "o-", color=ACCENT[2], linewidth=2.5, markersize=7)

    # Inflection between 8 MB and 32 MB
    ax.axvspan(8*1024, 32*1024, alpha=0.12, color=ACCENT[2])
    ax.annotate("L3 size ≈ 16–25 MB\n(L3 miss rate spikes here)",
                xy=(20*1024, 1.0), xytext=(50*1024, 0.3),
                arrowprops=dict(arrowstyle="->", color=ACCENT[2]),
                color=ACCENT[2], fontsize=9)

    # Mark known data points
    for sz_mb, label in [(8, "8 MB\n0.047%"), (32, "32 MB\n1.743%")]:
        row = sub[sub["size_bytes"] == sz_mb*1024*1024]
        if not row.empty:
            v = row["l3_pct"].iloc[0]
            ax.annotate(label, xy=(sz_mb*1024, v),
                        xytext=(sz_mb*1024 * 0.6, v + 0.2),
                        fontsize=8, color="#8b949e")

    ax.set_xscale("log")
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("L3 Miss Rate (%)")
    ax.set_title("L3 Miss Rate vs Array Size — Detecting L3 Cache Boundary", fontsize=12, pad=10)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    savefig(fig, "l3_inflection.png")

# ── 4. All strides on one plot — latency vs array size ────────────────────────
def plot_all_strides(df):
    strides_to_plot = [1, 2, 8, 32, 64, 256, 1024]
    colors = ACCENT + ["#d29922", "#388bfd"]

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, stride in enumerate(strides_to_plot):
        sub = df[df["stride"] == stride].sort_values("size_bytes")
        if sub.empty:
            continue
        label = f"stride={stride}"
        ax.plot(sub["size_bytes"] / 1024, sub["ns_per_access"],
                "o-", color=colors[i % len(colors)],
                linewidth=1.8, markersize=5, label=label)

    # Cache level boundaries
    for size_kb, label, color in [
        (32,     "L1d\n32KB",  "#58a6ff"),
        (256,    "L2\n256KB",  "#3fb950"),
        (25600,  "L3\n25MB",   "#ff7b72"),
    ]:
        ax.axvline(x=size_kb, linestyle="--", linewidth=1, color=color, alpha=0.6)
        ax.text(size_kb * 1.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 100,
                label, fontsize=7.5, color=color, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("Latency (ns/access)")
    ax.set_title("Access Latency vs Array Size — All Stride Values", fontsize=12, pad=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    savefig(fig, "all_strides_latency.png")

if __name__ == "__main__":
    print("Generating warmup plots from real EdLab data...")
    df = load_data()
    plot_stride_latency(df)
    plot_cache_levels(df)
    plot_l3_inflection(df)
    plot_all_strides(df)
    print(f"\nDone — plots in {PLOTDIR}/")
