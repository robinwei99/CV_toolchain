#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_aruco_image.py
Detect ArUco codes in a single image and draw their borders & IDs.

Usage:
python detect_aruco_image.py --image input.jpg --dict 4X4_50 --save out.jpg
"""

import argparse
from pathlib import Path
import cv2

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

def detect_markers(gray, dictionary):
    # New API (OpenCV >= 4.7) fallback to old API
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected

def main():
    ap = argparse.ArgumentParser(description="Detect ArUco codes in one image")
    ap.add_argument("--image", required=True, help="Path to JPG/PNG image")
    ap.add_argument("--dict", default="4X4_50", help="Aruco dictionary name (default: 4X4_50)")
    args = ap.parse_args()

    dictionary = get_dictionary(args.dict)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detect_markers(gray, dictionary)

    vis = img.copy()
    num = 0 if ids is None else len(ids)
    if num > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        print(f"[OK] Detected {num} marker(s): {ids.flatten().tolist()}")
    else:
        print("[INFO] No markers detected.")

    scale = 0.3
    vis_small = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    cv2.imshow("ArUco Detection", vis_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
