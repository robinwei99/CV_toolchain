#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_charuco_letter_pdf.py
Generates a ChArUco calibration board in Letter (8.5x11in) PDF format at real-world scale,
and also outputs a JSON file for direct use by downstream registration code
to construct a CharucoBoard.

Example:
python gen_charuco_letter_pdf.py --squares-x 5 --squares-y 7 --square-mm 30 \
  --marker-frac 0.7 --aruco-dict 4X4_50 --margin-mm 8 \
  --out charuco_5x7_30mm.pdf --json charuco_5x7_30mm.json
"""

import argparse
from io import BytesIO
from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter  # 612 x 792 pt
from reportlab.lib.utils import ImageReader

# -------------- Common ArUco dictionary mappings --------------
DICT_MAP = {
    "4X4_50":     cv2.aruco.DICT_4X4_50,
    "4X4_100":    cv2.aruco.DICT_4X4_100,
    "4X4_250":    cv2.aruco.DICT_4X4_250,
    "4X4_1000":   cv2.aruco.DICT_4X4_1000,
    "5X5_50":     cv2.aruco.DICT_5X5_50,
    "5X5_100":    cv2.aruco.DICT_5X5_100,
    "5X5_250":    cv2.aruco.DICT_5X5_250,
    "5X5_1000":   cv2.aruco.DICT_5X5_1000,
    "6X6_50":     cv2.aruco.DICT_6X6_50,
    "6X6_100":    cv2.aruco.DICT_6X6_100,
    "6X6_250":    cv2.aruco.DICT_6X6_250,
    "6X6_1000":   cv2.aruco.DICT_6X6_1000,
    "7X7_50":     cv2.aruco.DICT_7X7_50,
    "7X7_100":    cv2.aruco.DICT_7X7_100,
    "7X7_250":    cv2.aruco.DICT_7X7_250,
    "7X7_1000":   cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

def get_dictionary(name: str):
    key = name.upper()
    if key not in DICT_MAP:
        raise ValueError(f"Unknown dictionary {name}; available: {', '.join(DICT_MAP.keys())}")
    return cv2.aruco.getPredefinedDictionary(DICT_MAP[key]), int(DICT_MAP[key])

def create_charuco_board(nx, ny, square_len, marker_len, dictionary):
    # Construct a CharucoBoard (pixel units are used here just for image generation)
    try:
        board = cv2.aruco.CharucoBoard_create(nx, ny, float(square_len), float(marker_len), dictionary)
    except AttributeError:
        board = cv2.aruco.CharucoBoard((nx, ny), float(square_len), float(marker_len), dictionary)
    return board

def render_charuco(nx, ny, square_mm, marker_frac, dictionary, dpi_render, margin_mm, footer_text=None):
    """
    Render a grayscale image at high DPI and return the image along with
    physical size info and pixel metrics for PDF layout and JSON output.
    """
    px_per_mm = dpi_render / 25.4
    sq_px = int(round(square_mm * px_per_mm))
    mk_px = int(round(square_mm * marker_frac * px_per_mm))
    if sq_px <= 0 or mk_px <= 0:
        raise ValueError("square-mm or marker-frac too small, resulting in 0 pixels. Increase size or DPI.")

    board_px_w = nx * sq_px
    board_px_h = ny * sq_px
    margin_px = int(round(margin_mm * px_per_mm))

    board = create_charuco_board(nx, ny, square_len=sq_px, marker_len=mk_px, dictionary=dictionary)
    try:
        img = board.draw((board_px_w, board_px_h))
    except AttributeError:
        img = board.generateImage((board_px_w, board_px_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Add white margin around the board
    canvas = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px,
                                borderType=cv2.BORDER_CONSTANT, value=255)

    # Optional footer text (does not affect geometry)
    extra_footer_mm = 0.0
    if footer_text:
        pad_px = int(round(4 * px_per_mm))  # extra 4mm
        canvas = cv2.copyMakeBorder(canvas, 0, pad_px, 0, 0, cv2.BORDER_CONSTANT, value=255)
        y = canvas.shape[0] - int(round(1.5 * px_per_mm))
        cv2.putText(canvas, footer_text, (int(round(5 * px_per_mm)), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 1, cv2.LINE_AA)
        extra_footer_mm = 4.0

    # Physical total size (mm)
    total_w_mm = nx * square_mm + 2 * margin_mm
    total_h_mm = ny * square_mm + 2 * margin_mm + extra_footer_mm

    # Pixel info summary
    pixel_info = {
        "board_pixel_width": int(board_px_w),
        "board_pixel_height": int(board_px_h),
        "canvas_pixel_width": int(canvas.shape[1]),
        "canvas_pixel_height": int(canvas.shape[0]),
        "square_pixel": int(sq_px),
        "marker_pixel": int(mk_px),
        "margin_pixel": int(margin_px),
        "dpi_render": int(dpi_render)
    }
    return canvas, total_w_mm, total_h_mm, pixel_info

def place_on_letter_pdf(image_np, phys_w_mm, phys_h_mm, out_pdf, title_meta=None):
    """Place the image at the center of a Letter page with real-world dimensions."""
    page_w_pt, page_h_pt = letter  # (612, 792)
    page_w_mm = page_w_pt / 72.0 * 25.4
    page_h_mm = page_h_pt / 72.0 * 25.4

    if phys_w_mm > page_w_mm + 1e-6 or phys_h_mm > page_h_mm + 1e-6:
        raise ValueError(f"Board size exceeds Letter page: board {phys_w_mm:.1f}×{phys_h_mm:.1f} mm, "
                         f"page {page_w_mm:.1f}×{page_h_mm:.1f} mm. Reduce size.")

    ok, png_bytes = cv2.imencode(".png", image_np)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    bio = BytesIO(png_bytes.tobytes())
    img_reader = ImageReader(bio)

    w_pt = phys_w_mm / 25.4 * 72.0
    h_pt = phys_h_mm / 25.4 * 72.0
    x_pt = (page_w_pt - w_pt) / 2.0
    y_pt = (page_h_pt - h_pt) / 2.0

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    if title_meta:
        c.setTitle(title_meta)
        c.setAuthor("Charuco Generator")
        c.setSubject("Calibration Board")
        c.setCreator("gen_charuco_letter_pdf.py")
        c.setKeywords(["ChArUco", "Calibration", "OpenCV"])
    c.drawImage(img_reader, x_pt, y_pt, width=w_pt, height=h_pt, preserveAspectRatio=False, mask='auto')
    c.showPage()
    c.save()

def main():
    ap = argparse.ArgumentParser(description="Generate ChArUco (Letter PDF + JSON)")
    ap.add_argument("--squares-x", type=int, default=5, help="Number of inner corners (X direction)")
    ap.add_argument("--squares-y", type=int, default=7, help="Number of inner corners (Y direction)")
    ap.add_argument("--square-mm", type=float, default=30, help="Square side length (mm)")
    ap.add_argument("--dict", default="4X4_50", help="Aruco dictionary name, e.g., 4X4_50 / 5X5_100 / ARUCO_ORIGINAL")
    ap.add_argument("--marker-frac", type=float, default=0.7, help="marker side length = marker-frac * square-mm")
    ap.add_argument("--margin-mm", type=float, default=8.0, help="White margin around the board (mm)")
    ap.add_argument("--dpi-render", type=int, default=1200, help="Render DPI (600/1200 recommended)")
    ap.add_argument("--label", type=str, default=None, help="Optional footer text")
    ap.add_argument("--out", type=str, default=None, help="Output PDF path")
    args = ap.parse_args()

    nx, ny = int(args.squares_x), int(args.squares_y)
    dictionary, dict_id = get_dictionary(args.dict)

    footer = args.label
    if footer is None:
        footer = (f"ChArUco  {nx}x{ny}  square={args.square_mm}mm  "
                  f"dict={args.dict}  frac={args.marker_frac}  "
                  f"generated {datetime.now().strftime('%Y-%m-%d')}")

    img, total_w_mm, total_h_mm, pxinfo = render_charuco(
        nx, ny, args.square_mm, args.marker_frac, dictionary,
        dpi_render=args.dpi_render, margin_mm=args.margin_mm, footer_text=footer
    )

    if args.out != None:
        out_pdf = Path(args.out)
    else:
        out_pdf = Path("charuco_"+str(args.squares_x)+"x"+str(args.squares_y)+"_"+str(args.square_mm)+"mm.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    place_on_letter_pdf(
        img, total_w_mm, total_h_mm, out_pdf,
        title_meta=f"ChArUco {nx}x{ny} {args.square_mm}mm ({args.dict})"
    )

    # ---- JSON export (for direct use in registration code) ----
    out_json = out_pdf.with_suffix(".json")
    meta = {
        "board_type": "charuco",
        # Charuco topology (nx/ny are number of inner corners)
        "squares_x": nx,
        "squares_y": ny,
        # Physical dimensions (mm)
        "square_length_mm": float(args.square_mm),
        "marker_length_mm": float(args.square_mm * args.marker_frac),
        "margin_mm": float(args.margin_mm),
        "units": "mm",
        # Dictionary info
        "aruco_dict_name": args.dict,
        "aruco_dict_id": dict_id,  # for direct use with getPredefinedDictionary(id)
        # Page and placement
        "page": {
            "type": "letter",
            "width_mm": float(215.9),   # 8.5in
            "height_mm": float(279.4)   # 11in
        },
        # Actual board size in PDF (including white margin and footer space)
        "board_total_width_mm": float(total_w_mm),
        "board_total_height_mm": float(total_h_mm),
        # Render pixel info (for mm/px ratio computation)
        "render": pxinfo,  # contains dpi_render, canvas/board pixel sizes, square/marker pixels
        # Coordinate system and origin convention for generating 3D points
        "world_frame": {
            "origin": "top_left_chessboard_corner",
            "x_axis": "right",
            "y_axis": "down",
            "z_axis": "out_of_board_plane"
        },
        # Extra info
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": args.label
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] PDF generated: {out_pdf.resolve()}")
    print(f"[OK] JSON generated: {out_json.resolve()}")
    print("Printing tip: Make sure to choose 'Actual size / 100% / No scaling' in print settings.")

if __name__ == "__main__":
    main()
