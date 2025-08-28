#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot XYZ position (and optional 3D trajectory) from an ArUco poses CSV.

Expected CSV columns (default):
  frame,time_sec,marker_id,x,y,z,rx,ry,rz

Usage examples:
  # Basic: plot all markers' X/Y/Z vs time (seconds)
  python plot_aruco_csv.py --csv poses.csv

  # Filter to specific marker(s) and use frames as x-axis
  python plot_aruco_csv.py --csv poses.csv --markers 3 7 --xaxis frame

  # Smooth with moving average (window of 15 points), treat units as meters -> convert to m
  python plot_aruco_csv.py --csv poses.csv --unit m --smooth 15

  # Draw a 3D trajectory (per marker) and save the figure
  python plot_aruco_csv.py --csv poses.csv --save fig.png

Columns can be remapped via flags if your CSV headers differ.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    # pad with edge values to keep length
    pad = w // 2
    a_pad = np.pad(a, (pad, pad), mode="edge")
    kernel = np.ones(w) / float(w)
    return np.convolve(a_pad, kernel, mode="valid")

def parse_args():
    ap = argparse.ArgumentParser(description="Plot XYZ from ArUco poses CSV")
    ap.add_argument("--csv", required=True, help="Path to poses CSV")
    ap.add_argument("--xaxis", choices=["time","frame"], default="time",
                    help="Use 'time' (time_sec) or 'frame' as x-axis (default: time)")
    ap.add_argument("--markers", type=int, nargs="*", default=None,
                    help="Optional list of marker IDs to include (default: all)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Moving average window length (>=1, default 1=no smoothing)")
    ap.add_argument("--unit", default="mm",
                    help="Input unit for x,y,z in CSV (default: mm). Options: m,cm,mm,um,nm")
    ap.add_argument("--save", type=str, default=None,
                    help="If set, save the figure to this path instead of (or in addition to) showing")
    # Column name overrides
    ap.add_argument("--col-frame", default="frame")
    ap.add_argument("--col-time", default="time_sec")
    ap.add_argument("--col-id", default="marker_id")
    ap.add_argument("--col-x", default="x")
    ap.add_argument("--col-y", default="y")
    ap.add_argument("--col-z", default="z")
    return ap.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # verify required columns
    for col in [args.col_frame, args.col_time, args.col_id, args.col_x, args.col_y, args.col_z]:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)

    # filter markers
    if args.markers:
        df = df[df[args.col_id].isin(args.markers)]
        if df.empty:
            print("[WARN] No rows after filtering markers; check IDs.", file=sys.stderr)

    # choose x-axis
    if args.xaxis == "time":
        x = df[args.col_time].to_numpy()
        x_label = "Time (s)"
    else:
        x = df[args.col_frame].to_numpy()
        x_label = "Frame"

    # convert units if requested
    unit_label = args.unit

    # Prepare figure: top 3 subplots for X/Y/Z vs x-axis
    nrows = 3
    fig = plt.figure(figsize=(11, 9))

    # We will plot per marker group
    groups = list(df.groupby(args.col_id))

    # X
    ax1 = fig.add_subplot(nrows, 1, 1)
    for mid, g in groups:
        xs = moving_average((g[args.col_x].to_numpy()), max(1, args.smooth))
        ax1.plot(g[args.xaxis == "time" and args.col_time or args.col_frame], xs, label=f"Marker {mid}")
    ax1.set_ylabel(f"X ({unit_label})")
    ax1.grid(True)
    ax1.legend(loc="best")

    # Y
    ax2 = fig.add_subplot(nrows, 1, 2, sharex=ax1)
    for mid, g in groups:
        ys = moving_average((g[args.col_y].to_numpy()), max(1, args.smooth))
        ax2.plot(g[args.xaxis == "time" and args.col_time or args.col_frame], ys, label=f"Marker {mid}")
    ax2.set_ylabel(f"Y ({unit_label})")
    ax2.grid(True)

    # Z
    ax3 = fig.add_subplot(nrows, 1, 3, sharex=ax1)
    for mid, g in groups:
        zs = moving_average((g[args.col_z].to_numpy()), max(1, args.smooth))
        ax3.plot(g[args.xaxis == "time" and args.col_time or args.col_frame], zs, label=f"Marker {mid}")
    ax3.set_ylabel(f"Z ({unit_label})")
    ax3.set_xlabel(x_label)
    ax3.grid(True)

    fig.suptitle("ArUco Marker Position")
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=600)
        print(f"[INFO] Saved figure to: {out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()