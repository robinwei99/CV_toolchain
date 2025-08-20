#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_chessboard_letter_pdf.py
Generates a standard black–white chessboard (checkerboard) calibration target
in Letter (8.5x11in) PDF format at real-world scale, and outputs a JSON file
for downstream code to reconstruct the board geometry.

NOTES
- "squares-x"/"squares-y" are the number of INNER CORNERS used by OpenCV's
  findChessboardCorners, i.e., printed squares = inner corners + 1.
- The board will be centered on the page with an optional white margin.
- Top-left square is BLACK, alternating pattern.

Example:
python gen_chessboard_letter_pdf.py --squares-x 9 --squares-y 6 --square-mm 25 \
  --margin-mm 8 --dpi-render 1200 \
  --out chessboard_9x6_25mm.pdf
"""

import argparse
from io import BytesIO
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import cv2
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import letter  # 612 x 792 pt
from reportlab.lib.utils import ImageReader

# ---------------------------- Rendering ------------------------------------

def render_chessboard(nx_inner: int, ny_inner: int, square_mm: float,
                      dpi_render: int, margin_mm: float, footer_text: str | None):
    """
    Render a standard chessboard as a high-DPI grayscale image with a white
    margin and optional footer area.

    nx_inner, ny_inner: number of inner corners (OpenCV convention)
    square_mm: physical side length of one printed square (mm)
    dpi_render: DPI to rasterize the image before embedding into PDF
    margin_mm: white margin around the printed board (mm)

    Returns: (canvas_img, total_w_mm, total_h_mm, pixel_info)
    """
    if nx_inner <= 0 or ny_inner <= 0:
        raise ValueError("squares-x and squares-y must be positive inner-corner counts")
    if square_mm <= 0:
        raise ValueError("square-mm must be > 0")

    px_per_mm = dpi_render / 25.4

    # Printed squares = inner corners + 1
    nx_sq = nx_inner + 1
    ny_sq = ny_inner + 1

    sq_px = int(round(square_mm * px_per_mm))
    if sq_px <= 0:
        raise ValueError("square-mm too small for chosen DPI (result is 0 px per square)")

    board_px_w = nx_sq * sq_px
    board_px_h = ny_sq * sq_px
    margin_px = int(round(margin_mm * px_per_mm))

    # Create alternating black/white tile image; top-left is black
    board = np.full((board_px_h, board_px_w), 255, dtype=np.uint8)
    for j in range(ny_sq):
        y0, y1 = j * sq_px, (j + 1) * sq_px
        for i in range(nx_sq):
            x0, x1 = i * sq_px, (i + 1) * sq_px
            if (i + j) % 2 == 0:  # black
                board[y0:y1, x0:x1] = 0

    # Add white margin around the board
    canvas_img = cv2.copyMakeBorder(board, margin_px, margin_px, margin_px, margin_px,
                                    borderType=cv2.BORDER_CONSTANT, value=255)

    extra_footer_mm = 0.0
    if footer_text:
        pad_px = int(round(4 * px_per_mm))  # extra 4mm
        canvas_img = cv2.copyMakeBorder(canvas_img, 0, pad_px, 0, 0,
                                        borderType=cv2.BORDER_CONSTANT, value=255)
        y = canvas_img.shape[0] - int(round(1.5 * px_per_mm))
        cv2.putText(canvas_img, footer_text, (int(round(5 * px_per_mm)), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 1, cv2.LINE_AA)
        extra_footer_mm = 4.0

    total_w_mm = nx_sq * square_mm + 2 * margin_mm
    total_h_mm = ny_sq * square_mm + 2 * margin_mm + extra_footer_mm

    pixel_info = {
        "canvas_pixel_width": int(canvas_img.shape[1]),
        "canvas_pixel_height": int(canvas_img.shape[0]),
        "board_pixel_width": int(board_px_w),
        "board_pixel_height": int(board_px_h),
        "square_pixel": int(sq_px),
        "margin_pixel": int(margin_px),
        "dpi_render": int(dpi_render),
    }
    return canvas_img, total_w_mm, total_h_mm, pixel_info


def place_on_letter_pdf(image_np, phys_w_mm, phys_h_mm, out_pdf: Path, title_meta: str | None):
    """Place the image at the center of a Letter page with exact physical size."""
    page_w_pt, page_h_pt = letter  # (612, 792)
    page_w_mm = page_w_pt / 72.0 * 25.4
    page_h_mm = page_h_pt / 72.0 * 25.4

    if phys_w_mm > page_w_mm + 1e-6 or phys_h_mm > page_h_mm + 1e-6:
        raise ValueError(
            f"Board size exceeds Letter page: board {phys_w_mm:.1f}×{phys_h_mm:.1f} mm, "
            f"page {page_w_mm:.1f}×{page_h_mm:.1f} mm. Reduce size."
        )

    ok, png_bytes = cv2.imencode(".png", image_np)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    bio = BytesIO(png_bytes.tobytes())
    img_reader = ImageReader(bio)

    w_pt = phys_w_mm / 25.4 * 72.0
    h_pt = phys_h_mm / 25.4 * 72.0
    x_pt = (page_w_pt - w_pt) / 2.0
    y_pt = (page_h_pt - h_pt) / 2.0

    c = rl_canvas.Canvas(str(out_pdf), pagesize=letter)
    if title_meta:
        c.setTitle(title_meta)
        c.setAuthor("Chessboard Generator")
        c.setSubject("Calibration Board")
        c.setCreator("gen_chessboard_letter_pdf.py")
        c.setKeywords(["Chessboard", "Calibration", "OpenCV"])
    c.drawImage(img_reader, x_pt, y_pt, width=w_pt, height=h_pt, preserveAspectRatio=False, mask='auto')
    c.showPage()
    c.save()


# ------------------------------ CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate standard Chessboard (Letter PDF + JSON)")
    ap.add_argument("--squares-x", type=int, default=6,
                    help="Number of inner corners (X direction, e.g., 6)")
    ap.add_argument("--squares-y", type=int, default=9,
                    help="Number of inner corners (Y direction, e.g., 9)")
    ap.add_argument("--square-mm", type=float, default=25,
                    help="Square side length (mm) for one printed square")
    ap.add_argument("--margin-mm", type=float, default=8.0,
                    help="White margin around the board (mm)")
    ap.add_argument("--dpi-render", type=int, default=1200,
                    help="Render DPI (600/1200 recommended)")
    ap.add_argument("--out", type=str, default=None, help="Output PDF path")

    args = ap.parse_args()

    nx_in, ny_in = int(args.squares_x), int(args.squares_y)

    footer = (f"Chessboard  inner={nx_in}x{ny_in}  square={args.square_mm}mm  "
                f"generated {datetime.now().strftime('%Y-%m-%d')}")

    img, total_w_mm, total_h_mm, pxinfo = render_chessboard(
        nx_in, ny_in, args.square_mm, dpi_render=args.dpi_render,
        margin_mm=args.margin_mm, footer_text=footer
    )

    if args.out is not None:
        out_pdf = Path(args.out)
    else:
        out_pdf = Path(f"chessboard_{nx_in}x{ny_in}_{int(args.square_mm)}mm.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    place_on_letter_pdf(
        img, total_w_mm, total_h_mm, out_pdf,
        title_meta=f"Chessboard inner {nx_in}x{ny_in} {args.square_mm}mm"
    )

    # ---- JSON export ----
    out_json = out_pdf.with_suffix(".json")
    meta = {
        "board_type": "chessboard",
        # Chessboard topology (OpenCV inner-corner counts)
        "squares_x": nx_in,
        "squares_y": ny_in,
        # Physical dimensions (mm)
        "square_length_mm": float(args.square_mm),
        "margin_mm": float(args.margin_mm),
        "units": "mm",
        # Page info
        "page": {
            "type": "letter",
            "width_mm": float(215.9),  # 8.5in
            "height_mm": float(279.4)  # 11in
        },
        # Actual board size in PDF (including white margin and footer space)
        "board_total_width_mm": float(total_w_mm),
        "board_total_height_mm": float(total_h_mm),
        # Render pixel info (for mm/px ratio computation)
        "render": pxinfo,
        # Coordinate system and origin convention for generating 3D points
        "world_frame": {
            "origin": "top_left_chessboard_corner",
            "x_axis": "right",
            "y_axis": "down",
            "z_axis": "out_of_board_plane"
        },
        # Extra info
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": footer,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] PDF generated: {out_pdf.resolve()}")
    print(f"[OK] JSON generated: {out_json.resolve()}")
    print("Printing tip: Choose 'Actual size / 100% / No scaling' in print settings.")


if __name__ == "__main__":
    main()
