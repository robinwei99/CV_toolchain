#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect ArUco codes in every frame of a video, estimate each tag's 3D pose,
optionally overlay axes, and write poses to a CSV file.

Usage examples:
  python detect_aruco_video.py \
      --video input.mp4 \
      --dict 4X4_50 \
      --length 0.04 \
      --camera calib.npz \
      --csv poses.csv \

The calib.npz file should contain 'camera_matrix' (3x3) and 'dist_coeffs' (1x5 or 1x8),
for example created by OpenCV's calibration tools.

Output CSV columns:
  frame,time_sec,marker_id,x,y,z,rx,ry,rz
where (x,y,z) are in meters in the camera coordinate system and (rx,ry,rz)
are Rodrigues rotation vector components (radians).
"""

import argparse
from pathlib import Path
import csv
import sys
import cv2
import numpy as np

# ---- ArUco dictionaries map ----
DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250, "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50, "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250, "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50, "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250, "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50, "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250, "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def get_dictionary(name: str):
    key = name.upper()
    if key not in DICT_MAP:
        raise ValueError(f"Unknown dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(DICT_MAP[key])


def create_detector(dictionary):
    """Return a detector compatible with both new and old OpenCV APIs."""
    try:
        # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        def _detect(gray):
            return detector.detectMarkers(gray)
        return _detect
    except AttributeError:
        # Older API
        params = cv2.aruco.DetectorParameters_create()
        def _detect(gray):
            return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return _detect


def estimate_pose_single_markers(corners, length, camera_matrix, dist_coeffs):
    """Compatibility wrapper for estimatePoseSingleMarkers across OpenCV versions."""
    # corners: list of (1,4,2) points or (4,2) depending on API
    try:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, length, camera_matrix, dist_coeffs
        )
        # Some versions return rvecs,tvecs with shape (N,1,3); squeeze to (N,3)
        rvecs = np.squeeze(rvecs)
        tvecs = np.squeeze(tvecs)
        if rvecs.ndim == 1:
            rvecs = rvecs[np.newaxis, :]
            tvecs = tvecs[np.newaxis, :]
        return rvecs, tvecs
    except AttributeError:
        # Fallback: manual solvePnP for each marker
        rvecs = []
        tvecs = []
        # define object points for square marker in its own coordinate frame
        half = length / 2.0
        objp = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)
        for c in corners:
            pts2d = np.array(c).reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, pts2d, camera_matrix, dist_coeffs)
            if not ok:
                rvec = np.full((3,), np.nan, dtype=np.float32)
                tvec = np.full((3,), np.nan, dtype=np.float32)
            rvecs.append(rvec.reshape(3))
            tvecs.append(tvec.reshape(3))
        return np.array(rvecs), np.array(tvecs)


def draw_pose(img, corners, rvecs, tvecs, camera_matrix, dist_coeffs, axis_len=0.75):
    for c, r, t in zip(corners, rvecs, tvecs):
        try:
            cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, r, t, axis_len)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Detect ArUco codes per video frame and log poses to CSV")
    ap.add_argument("--video", required=True, help="Path to an input video file (or integer camera index)")
    ap.add_argument("--dict", default="4X4_50", help="ArUco dictionary name (default: 4X4_50)")
    ap.add_argument("--length", type=float, required=True, help="Marker side length in meters (for pose)")
    ap.add_argument("--camera", type=str, required=True, help="Path to calib .npz with cameraMatrix & distCoeffs")
    ap.add_argument("--csv", type=str, default="poses.csv", help="Output CSV path")
    ap.add_argument("--exam", action="store_true", help="Write annotated video to exam")
    args = ap.parse_args()

    # Load calibration
    calib_path = Path(args.camera)
    if not calib_path.exists():
        raise SystemExit(f"Calibration file not found: {calib_path}")
    calib = np.load(str(calib_path))
    if "cameraMatrix" not in calib or "distCoeffs" not in calib:
        raise SystemExit("Calibration .npz must contain 'cameraMatrix' and 'distCoeffs'")
    camera_matrix = calib["cameraMatrix"]
    dist_coeffs = calib["distCoeffs"]

    # Video capture
    video_source = args.video
    try:
        # allow integer camera index
        video_source = int(video_source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0  # fallback
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional video writer
    writer = None
    if args.exam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("exam.mp4", fourcc, fps, (width, height))
        if not writer.isOpened():
            print("[WARN] Could not open video writer", file=sys.stderr)
            writer = None

    dictionary = get_dictionary(args.dict)
    detect_fn = create_detector(dictionary)

    # Prepare CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_sec", "marker_id", "x", "y", "z", "rx", "ry", "rz"])

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detection
            corners, ids, rejected = detect_fn(gray)

            if ids is not None and len(ids) > 0:
                # draw outlines for visibility
                try:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                except Exception:
                    pass

                # pose estimation for each marker
                rvecs, tvecs = estimate_pose_single_markers(corners, args.length, camera_matrix, dist_coeffs)

                # log to CSV
                t_sec = frame_idx / fps
                for i, mid in enumerate(ids.flatten().tolist()):
                    t = tvecs[i].astype(float)
                    r = rvecs[i].astype(float)
                    csv_writer.writerow([
                        frame_idx, f"{t_sec:.6f}", int(mid),
                        f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                        f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}",
                    ])

                # draw axes on the frame (optional visualization)
                draw_pose(frame, corners, rvecs, tvecs, camera_matrix, dist_coeffs, axis_len=args.length * 0.75)

            if writer is not None:
                writer.write(frame)

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()


if __name__ == "__main__":
    main()
