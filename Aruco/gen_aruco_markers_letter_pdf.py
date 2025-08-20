#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_aruco_markers_letter_pdf_blacksize.py

Generate multiple ArUco markers for a Letter-size PDF (8.5x11in), where the input
size (--code-mm) is the physical size of the *black code area only* (excluding the
quiet white border). The generator automatically adds a 1-module quiet zone.
A fixed 2 cm gap is used between markers for easy cutting (customizable via --gap-mm).

Print with "Actual size / 100% / No scaling".

Example:
python gen_aruco_markers_letter_pdf_blacksize.py \
  --dict 4X4_50 --code-mm 40 --count 12 --start-id 0 \
  --margin-mm 10 --gap-mm 20 --cols auto --dpi-render 600 \
  --out aruco_4x4_50_40mm_blackonly.pdf
"""

import argparse
from io import BytesIO
from pathlib import Path
import re

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter      # 612 x 792 pt
from reportlab.lib.utils import ImageReader

# -------- Common ArUco dictionary mapping --------
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

def mm_to_pt(mm_val: float) -> float:
    """Convert millimeters to PDF points."""
    return mm_val / 25.4 * 72.0

def mm_to_px(mm_val: float, dpi: float) -> int:
    """Convert millimeters to pixels for an offscreen render at given DPI."""
    return int(round(mm_val * dpi / 25.4))

def get_dictionary(name: str):
    """Get OpenCV ArUco dictionary object from string name."""
    key = name.upper()
    if key not in DICT_MAP:
        raise ValueError(f"Unknown dictionary {name}; available: {', '.join(DICT_MAP.keys())}")
    return cv2.aruco.getPredefinedDictionary(DICT_MAP[key])

def infer_module_count(dict_name: str) -> int:
    """
    Infer the number of modules per side (n) from the dictionary name.
    E.g., "4X4_50" -> 4, "6X6_250" -> 6, "ARUCO_ORIGINAL" -> assume 5.
    """
    dn = dict_name.upper()
    m = re.match(r"^([4-7])X\1_", dn)
    if m:
        return int(m.group(1))
    if dn == "ARUCO_ORIGINAL":
        return 5  # Original ArUco markers are 5x5
    raise ValueError(f"Cannot infer module count from dictionary '{dict_name}'")

def render_marker_png_blacksize(dict_obj, dict_name: str, marker_id: int,
                                code_mm: float, dpi_render: int) -> bytes:
    """
    Render a single ArUco marker PNG where 'code_mm' is the physical size
    of the black code area (excluding quiet zone). We add a 1-module quiet zone
    around it via 'border_bits'.

    Steps:
      - Compute module count 'n' from dictionary.
      - Choose an integer module size in pixels so that black area = n * module_px
        closely matches code_mm at given dpi (favor exact module alignment).
      - Total side in pixels = (n + 2*border_bits) * module_px.
    """
    # Desired black size in pixels:
    black_px_target = mm_to_px(code_mm, dpi_render)

    # Generate into a square image of size=side_px so quiet zone is included
    img = np.zeros((black_px_target, black_px_target), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dict_obj, int(marker_id), black_px_target, img)
    ok, png_bytes = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return png_bytes.tobytes()

def layout_grid( count, page_w_mm, page_h_mm,
                margin_mm, code_mm, gap_mm):
    """
    Compute rows, columns, and pages given layout constraints.

    code_mm: full physical size of aruco marker.
    gap_mm: fixed physical gap between tiles (e.g., 20mm = 2cm).
    """
    avail_w = page_w_mm - 2*margin_mm
    avail_h = page_h_mm - 2*margin_mm
    if avail_w < code_mm or avail_h < code_mm:
        raise ValueError("Margin too large; no usable area remains.")

    cols = int((avail_w + gap_mm) // (code_mm + gap_mm))
    rows = int((avail_h + gap_mm) // (code_mm + gap_mm))
    per_page = rows * cols

    pages = (count + per_page - 1) // per_page
    return rows, cols, per_page, pages

def main():
    ap = argparse.ArgumentParser(
        description="Generate multiple ArUco markers to Letter-size PDF")
    ap.add_argument("--dict", default="4X4_50",
                    help="Aruco dictionary: 4X4_50 / 5X5_100 / 6X6_250 / 7X7_1000 / ARUCO_ORIGINAL")
    ap.add_argument("--code-mm", type=float, default=3,
                    help="Physical size (mm) of the BLACK code area")
    ap.add_argument("--count", type=int, default=10, help="Number of markers")
    ap.add_argument("--start-id", type=int, default=0, help="Start ID (increment sequentially)")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="Page margin (mm)")
    ap.add_argument("--gap-mm", type=float, default=20.0,
                    help="Gap between markers (mm). Default 20mm = 2cm for cutting.")
    ap.add_argument("--dpi-render", type=int, default=1200, help="Offscreen render DPI (600/1200 recommended)")
    ap.add_argument("--label", default=True, help="Print ID label under each marker")
    ap.add_argument("--out", default="10_10mm_aruco_4X4_50.pdf", help="Output PDF path")
    args = ap.parse_args()

    # Prepare IDs
    id_list = [int(args.start_id) + i for i in range(int(args.count))]

    # Letter page dimensions in mm
    page_w_pt, page_h_pt = letter
    page_w_mm = page_w_pt / 72.0 * 25.4
    page_h_mm = page_h_pt / 72.0 * 25.4

    # Derive full tile size (including quiet zone) from black code size and module count
    n = infer_module_count(args.dict)
    # tile_mm = args.code_mm * (1.0 + 2.0 / n)  # black + 1-module quiet zone on each side

    # Compute layout
    rows, cols, per_page, pages = layout_grid(
        count=len(id_list),
        page_w_mm=page_w_mm, page_h_mm=page_h_mm,
        margin_mm=args.margin_mm, code_mm=args.code_mm, gap_mm=args.gap_mm
    )

    # Prepare PDF canvas
    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    c.setTitle(f"Aruco {args.dict} code={args.code_mm}mm (black) x{args.count}")

    # Unit conversions to points
    code_pt = mm_to_pt(args.code_mm)
    gap_pt = mm_to_pt(args.gap_mm)
    margin_pt = mm_to_pt(args.margin_mm)

    usable_x0 = margin_pt
    usable_y0 = margin_pt
    usable_w = page_w_pt - 2 * margin_pt
    usable_h = page_h_pt - 2 * margin_pt

    dict_obj = get_dictionary(args.dict)

    # Draw pages
    idx = 0
    for p in range(pages):
        grid_w = cols * code_pt + (cols - 1) * gap_pt
        grid_h = rows * code_pt + (rows - 1) * gap_pt
        ox = usable_x0 + (usable_w - grid_w) / 2.0
        oy = usable_y0 + (usable_h - grid_h) / 2.0

        for r in range(rows):
            for col in range(cols):
                if idx >= len(id_list):
                    break
                marker_id = id_list[idx]

                # Render with black-size calibration so printed black area = code-mm
                png_bytes = render_marker_png_blacksize(
                    dict_obj=dict_obj,
                    dict_name=args.dict,
                    marker_id=marker_id,
                    code_mm=args.code_mm,
                    dpi_render=args.dpi_render,
                )
                img_reader = ImageReader(BytesIO(png_bytes))

                x = ox + col * (code_pt + gap_pt)
                y = oy + (rows - 1 - r) * (code_pt + gap_pt)

                # Draw full tile (black + quiet zone) to exact physical size
                c.drawImage(img_reader, x, y, width=code_pt, height=code_pt,
                            preserveAspectRatio=False, mask='auto')

                if args.label:
                    c.setFont("Helvetica", 8)
                    c.drawCentredString(x + code_pt / 2.0, y - mm_to_pt(args.gap_mm/2), f"id={marker_id}")

                idx += 1
            if idx >= len(id_list):
                break

        c.showPage()

    c.save()
    print(f"[OK] PDF generated: {out_pdf.resolve()}")
    print("Printing tip: Choose 'Actual size / 100% / No scaling' in print settings.")
    
if __name__ == "__main__":
    main()
