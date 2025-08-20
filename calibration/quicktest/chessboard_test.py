#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, cv2

def main():
    ap = argparse.ArgumentParser(description="Minimal chessboard corner viewer (no save)")
    ap.add_argument("--img", required=True, help="input image path")
    ap.add_argument("--nx", type=int, default=6, help="inner corners along x (e.g., 6)")
    ap.add_argument("--ny", type=int, default=9, help="inner corners along y (e.g., 9)")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"[ERR] cannot read: {args.img}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (31,31), 0)
    cv2.namedWindow("chessboard corners", cv2.WINDOW_NORMAL)
    cv2.imshow("chessboard corners", gray)
    cv2.resizeWindow("chessboard corners", 1280, 720)
    cv2.waitKey(0)
    pattern = (args.nx, args.ny)

    # 尽量简单：能用 SB 就用 SB，否则退回经典
    ok, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        except TypeError:
            pass
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    if not ok or corners is None:
        raise SystemExit("[FAIL] chessboard NOT found")

    # 画角点并显示
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, True)
    cv2.namedWindow("chessboard corners", cv2.WINDOW_NORMAL)
    cv2.imshow("chessboard corners", vis)
    cv2.resizeWindow("chessboard corners", 1280, 720)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
