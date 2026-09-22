"""
Plot all training_points and extra_training_points for 2018 and 2024.

Usage:
    python scripts/plot_training_points.py
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config import (
    POINTS_2018_CSV, POINTS_2024_CSV,
    EXTRA_POINTS_2018_CSV, EXTRA_POINTS_2024_CSV,
    CLASS_NAMES, CLASS_COLORS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_geo(geo_str):
    """Extract (lon, lat) from a GEE-style GeoJSON string."""
    g = json.loads(geo_str)
    gtype = g["type"]
    if gtype == "Point":
        return g["coordinates"][0], g["coordinates"][1]
    elif gtype in ("Polygon", "MultiPolygon"):
        if gtype == "Polygon":
            rings = g["coordinates"]
        else:
            rings = [r for poly in g["coordinates"] for r in poly]
        pts = np.array([pt for ring in rings for pt in ring])
        return pts[:, 0].mean(), pts[:, 1].mean()
    else:
        raise ValueError(f"Unsupported geometry type: {gtype}")


def load_points(csv_path):
    """Load a training-points CSV and return a DataFrame with lon, lat, label."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns and "class" in df.columns:
        df["label"] = df["class"]
    coords = df[".geo"].apply(parse_geo)
    df["lon"] = coords.apply(lambda c: c[0])
    df["lat"] = coords.apply(lambda c: c[1])
    return df[["lon", "lat", "label"]]


def plot_year(ax, tp_df, ep_df, year, class_names, class_colors):
    """Plot training + extra points for one year on an Axes."""
    for label_id, (name, color) in enumerate(zip(class_names, class_colors)):
        mask_tp = tp_df["label"] == label_id
        mask_ep = ep_df["label"] == label_id
        ax.scatter(
            tp_df.loc[mask_tp, "lon"], tp_df.loc[mask_tp, "lat"],
            c=color, s=18, marker="o", alpha=0.75, edgecolors="k",
            linewidths=0.3, label=f"{name} (training)",
        )
        ax.scatter(
            ep_df.loc[mask_ep, "lon"], ep_df.loc[mask_ep, "lat"],
            c=color, s=10, marker="^", alpha=0.55,
            linewidths=0, label=f"{name} (extra)",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Training & Extra Points — {year}", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def build_legend(class_names, class_colors):
    """Build a combined legend with class colors and source markers."""
    handles = []
    for name, color in zip(class_names, class_colors):
        handles.append(mpatches.Patch(facecolor=color, edgecolor="k", linewidth=0.5, label=name))
    handles.append(Line2D([], [], marker="o", color="grey", markeredgecolor="k",
                          markersize=6, linestyle="None", label="Training points"))
    handles.append(Line2D([], [], marker="^", color="grey",
                          markersize=6, linestyle="None", label="Extra points"))
    return handles


def main():
    os.chdir(PROJECT_ROOT)

    print("Loading CSVs …")
    tp_2018 = load_points(POINTS_2018_CSV)
    tp_2024 = load_points(POINTS_2024_CSV)
    ep_2018 = load_points(EXTRA_POINTS_2018_CSV)
    ep_2024 = load_points(EXTRA_POINTS_2024_CSV)

    for tag, df in [("training_2018", tp_2018), ("extra_2018", ep_2018),
                    ("training_2024", tp_2024), ("extra_2024", ep_2024)]:
        print(f"  {tag:20s}: {len(df):>6,d} points, "
              f"labels {sorted(df['label'].unique())}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_year(axes[0], tp_2018, ep_2018, 2018, CLASS_NAMES, CLASS_COLORS)
    plot_year(axes[1], tp_2024, ep_2024, 2024, CLASS_NAMES, CLASS_COLORS)

    legend_handles = build_legend(CLASS_NAMES, CLASS_COLORS)
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(CLASS_NAMES) + 2,
               fontsize=9, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("LULC Training Points Overview", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = PROJECT_ROOT / "outputs" / "training_points_map.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
