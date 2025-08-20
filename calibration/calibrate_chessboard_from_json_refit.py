#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_chessboard_from_json.py (modified)
- Performs standard chessboard calibration
- Shows per-image RMS reprojection error as a plot
"""

import argparse
import json
from pathlib import Path
import sys
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------ I/O -----------------------------------------

def load_config(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("board_type", "").lower() != "chessboard":
        raise ValueError("JSON is not chessboard type (board_type != 'chessboard').")
    req = ["squares_x", "squares_y", "square_length_mm"]
    for k in req:
        if k not in cfg:
            raise KeyError(f"JSON missing required field: {k}")
    return cfg


def collect_images(folder: str):
    exts = ("*.jpg","*.JPG","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(str(Path(folder) / e))))
    return paths

# --------------------------- Chessboard utils -------------------------------

def make_object_points(nx_inner: int, ny_inner: int, square_mm: float):
    objp = np.zeros((ny_inner * nx_inner, 3), np.float32)
    grid = np.mgrid[0:nx_inner, 0:ny_inner].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_mm)
    return objp


def detect_chessboard(gray, pattern_size, use_classic=False):
    if not use_classic and hasattr(cv2, "findChessboardCornersSB"):
        try:
            ok, corners = cv2.findChessboardCornersSB(
                gray, pattern_size,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
            if ok:
                return True, corners
        except TypeError:
            pass
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    return ok, corners


def subpix_refine(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    win = (11, 11)
    zero_zone = (-1, -1)
    cv2.cornerSubPix(gray, corners, win, zero_zone, criteria)
    return corners

# --------------------------- Error metrics ----------------------------------

def mean_reproj_error(objpoints_list, imgpoints_list, rvecs, tvecs, K, D):
    tot_err2 = 0.0
    tot_pts = 0
    for objp, imgp, rvec, tvec in zip(objpoints_list, imgpoints_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        err = cv2.norm(imgp, proj, cv2.NORM_L2)
        n = len(objp)
        tot_err2 += (err * err)
        tot_pts += n
    if tot_pts == 0:
        return float("nan")
    return float(np.sqrt(tot_err2 / tot_pts))

def per_image_errors(objpoints_list, imgpoints_list, rvecs, tvecs, K, D):
    per_img_rms = []
    for objp, imgp, rvec, tvec in zip(objpoints_list, imgpoints_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        err = cv2.norm(imgp, proj, cv2.NORM_L2) / np.sqrt(len(objp))
        per_img_rms.append(err)
    return per_img_rms

# ------------------------------- Main ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Camera calibration with a standard chessboard (from JSON)")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--use_classic", action="store_true")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--save", default="calib.npz")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    nx, ny = int(cfg["squares_x"]), int(cfg["squares_y"])
    square_mm = float(cfg["square_length_mm"])
    pattern_size = (nx, ny)
    objp = make_object_points(nx, ny, square_mm)

    img_paths = collect_images(args.folder)
    if not img_paths:
        print(f"[ERR] No images in {args.folder}")
        sys.exit(1)

    objpoints, imgpoints = [], []
    imsize, used = None, 0

    valid_paths = []

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (31,31), 0)
        if imsize is None:
            imsize = gray.shape[::-1]

        ok, corners = detect_chessboard(gray, pattern_size, use_classic=args.use_classic)
        if not ok:
            print(f"[WARN] {Path(p).name}: chessboard NOT found")
            continue

        if args.use_classic or not hasattr(cv2, "findChessboardCornersSB"):
            corners = subpix_refine(gray, corners)

        if corners is None or len(corners) != nx * ny:
            print(f"[WARN] {Path(p).name}: invalid corner count")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        valid_paths.append(Path(p).name)
        used += 1

    if used < max(args.min_frames, 3):
        print(f"[ERR] Only {used} valid detections")
        sys.exit(2)

    print(f"[INFO] calibrating with {used} images")
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)

    mean_err = mean_reproj_error(objpoints, imgpoints, rvecs, tvecs, K, D)
    print(f"[RESULT] Global RMS = {rms:.6f}, mean reproj error = {mean_err:.6f}")

    # Per-image RMS
    per_img_rms = per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, D)
    for name, err in zip(valid_paths, per_img_rms):
        print(f"[PER-IMG] {name}: RMS = {err:.4f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.bar(range(len(per_img_rms)), per_img_rms, tick_label=valid_paths)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMS reprojection error (px)")
    plt.title("Per-image RMS errors")
    plt.tight_layout()
    plt.show()

    # Save
    out_npz = Path(args.save)
    np.savez(out_npz,
             cameraMatrix=K,
             distCoeffs=D,
             imageSize=np.array(imsize, dtype=np.int32),
             rms=float(rms),
             reproj_error=float(mean_err),
             method="Chessboard",
             config=Path(args.cfg).name)
    print(f"[OK] Saved to {out_npz.resolve()}")

if __name__ == "__main__":
    main()
