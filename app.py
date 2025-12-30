import os
import io
import cv2
import time
import json
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------------------------------------
# Utilities
# ---------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def save_image(path: Path, image: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])

def zip_bytes_from_dict(filedict: Dict[str, bytes]) -> bytes:
    """
    Create an in-memory zip from a dict mapping filename->bytes and return bytes.
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in filedict.items():
            zf.writestr(fname, data)
    bio.seek(0)
    return bio.read()

# ---------------------------------------
# EL-specific preprocessing
# ---------------------------------------
def normalize_el(img_bgr: np.ndarray, clahe_clip: float = 2.5, tile: int = 8, blur_ksize: int = 3) -> np.ndarray:
    """
    Normalize EL image to enhance gridlines/busbars.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    gray_norm = clahe.apply(gray)
    if blur_ksize and blur_ksize > 0:
        blur_k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray_norm = cv2.GaussianBlur(gray_norm, (blur_k, blur_k), 0)
    return gray_norm

def auto_deskew(img_bgr: np.ndarray, gray: np.ndarray, hough_thresh: int = 120) -> np.ndarray:
    """
    Estimate dominant angle of lines and rotate to align vertical/horizontal grid.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        return img_bgr
    angles = []
    for l in lines[:400]:
        theta = l[0][1]
        deg = np.rad2deg(theta)
        # Map to [-90, 90)
        deg = ((deg + 90) % 180) - 90
        angles.append(deg)
    if len(angles) == 0:
        return img_bgr
    # Take median (more robust to outliers)
    mean_angle = float(np.median(angles))
    if abs(mean_angle) < 0.25:  # essentially no rotation
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -mean_angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def perspective_warp(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Try to find module contour (largest quadrilateral) and warp to a rectangle.
    Returns (warped_img, M, Minv). If detection fails, returns (img_bgr, I, I).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        I = np.eye(3, dtype=np.float32)
        return img_bgr, I, I
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        I = np.eye(3, dtype=np.float32)
        return img_bgr, I, I
    pts = approx.reshape(4, 2).astype(np.float32)
    # order points tl, tr, br, bl
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(1, max(widthA, widthB)))
    maxH = int(max(1, max(heightA, heightB)))
    # reasonable size guard
    if maxW < 50 or maxH < 50:
        I = np.eye(3, dtype=np.float32)
        return img_bgr, I, I
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    Minv = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# ---------------------------------------
# Grid line detection + cell building
# ---------------------------------------
def detect_grid_lines(gray: np.ndarray,
                      polarity: str = "auto",
                      binarize: str = "otsu",
                      ksize_v: int = 25,
                      ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vertical/horizontal line maps using morphology.
    polarity: 'auto' | 'dark' | 'bright'
    binarize: 'otsu' | 'adaptive'
    """
    if binarize == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Decide whether busbars/lines are dark or bright
    if polarity == "auto":
        use = 255 - bw if np.mean(gray) > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw

    # Vertical lines extraction
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, ksize_v)))
    vert = cv2.erode(use, kernel_v, iterations=1)
    vert = cv2.dilate(vert, kernel_v, iterations=1)

    # Horizontal lines extraction
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, ksize_h), 1))
    horiz = cv2.erode(use, kernel_h, iterations=1)
    horiz = cv2.dilate(horiz, kernel_h, iterations=1)

    # small closing to join fragments
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, small)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, small)

    return vert, horiz

def project_peaks(line_map: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    """
    Find peaks along projections of line_map.
    axis=0 -> sum over rows -> per-column (vertical)
    axis=1 -> sum over columns -> per-row (horizontal)
    """
    proj = line_map.sum(axis=axis).astype(np.float32)
    if proj.max() <= 0:
        return []
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min())
    peaks = []
    last_idx = -min_dist
    L = len(proj_norm)
    for i in range(1, L - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    """
    Convert peak positions (lines) to boundary cuts between cells.
    """
    if len(peaks) < 2:
        return [0, maxlen]
    cuts = [0]
    for i in range(len(peaks) - 1):
        cuts.append((peaks[i] + peaks[i + 1]) // 2)
    cuts.append(maxlen)
    # Ensure sorted, unique
    return sorted(list(dict.fromkeys(cuts)))

def grid_cells_from_maps(img_bgr: np.ndarray,
                          vert_map: np.ndarray,
                          horiz_map: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    H, W = vert_map.shape
    # vertical peaks (columns) -> axis=0
    xs = project_peaks(vert_map, axis=0, min_dist=max(10, W//40), min_strength=0.15)
    ys = project_peaks(horiz_map, axis=1, min_dist=max(10, H//40), min_strength=0.15)

    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)

    cells = []
    for r in range(len(ycuts)-1):
        y0, y1 = ycuts[r], ycuts[r+1]
        for c in range(len(xcuts)-1):
            x0, x1 = xcuts[c], xcuts[c+1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_bgr[y0:y1, x0:x1].copy()
                cells.append({
                    "row": r,
                    "col": c,
                    "bbox_warp": (x0, y0, x1, y1),
                    "image_warp": crop
                })
    return cells

def warp_bbox_to_original(bbox_warp: Tuple[int,int,int,int], Minv: np.ndarray, clip_shape: Tuple[int,int]) -> Tuple[int,int,int,int]:
    """
    Map bbox corners from warped plane back to the original image using inverse transform Minv.
    Returns axis-aligned bbox in original image coords (x, y, w, h).
    """
    x0, y0, x1, y1 = bbox_warp
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32).reshape(-1,1,2)
    if Minv is None:
        # identity mapping
        pts = corners.reshape(-1,2)
    else:
        pts = cv2.perspectiveTransform(corners, Minv).reshape(-1,2)
    xs = pts[:,0]
    ys = pts[:,1]
    xi0 = int(max(0, np.floor(xs.min())))
    yi0 = int(max(0, np.floor(ys.min())))
    xi1 = int(min(clip_shape[1], np.ceil(xs.max())))
    yi1 = int(min(clip_shape[0], np.ceil(ys.max())))
    if xi1 <= xi0 or yi1 <= yi0:
        return xi0, yi0, 0, 0
    return xi0, yi0, xi1 - xi0, yi1 - yi0

def build_mask_for_bbox(warped_gray: np.ndarray, bbox_warp: Tuple[int,int,int,int]) -> np.ndarray:
    """
    Build a binary mask for the bbox region from the warped grayscale image.
    """
    x0, y0, x1, y1 = bbox_warp
    crop = warped_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((0,0), dtype=np.uint8)
    # local Otsu then morphological clean
    _, m = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure mask corresponds to cells (cells often bright -> may need invert)
    # We keep mask as foreground where brighter than background
    k = max(1, min(7, (min(crop.shape)//20)|1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return (m > 0).astype(np.uint8)

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="PV EL Module → Cell Segregation", layout="wide")
st.title("🔬 PV EL Module → Cell Segregation (Streamlit)")

st.markdown("""
Upload **EL PV module** images to automatically detect the **cell grid** and export **per-cell crops**.
If auto detection struggles, use the **Manual rows/columns** fallback.
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# Preprocessing
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 4.0, 2.5, 0.1)
clahe_tile = st.sidebar.slider("CLAHE tile size", 4, 16, 8, 1)
blur_ksize = st.sidebar.slider("Gaussian blur (ksize)", 0, 7, 3, 1)

# Orientation correction
do_warp = st.sidebar.checkbox("Perspective warp (try to rectangularize)", True)
do_deskew = st.sidebar.checkbox("Auto deskew (align grid)", True)

# Detection params
polarity = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

# Fallback manual split
use_manual = st.sidebar.checkbox("Use manual rows × cols fallback", False)
n_rows = st.sidebar.number_input("Rows", min_value=1, max_value=20, value=6)
n_cols = st.sidebar.number_input("Cols", min_value=1, max_value=24, value=10)
manual_margin = st.sidebar.number_input("Manual margin (px)", min_value=0, max_value=200, value=0)

# Output options
out_dir_str = st.sidebar.text_input("Output directory (optional, also in-memory)", "output")
start_btn = st.sidebar.button("🚀 Run")

# Uploader
uploads = st.file_uploader("Upload EL module image(s)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

# ---------------------------------------
# Processing
# ---------------------------------------
def process_single_image(img_pil: Image.Image,
                         settings: Dict[str, Any],
                         save_root: Path = None) -> Dict[str, Any]:
    t0 = time.time()
    img_orig = pil_to_cv(img_pil)  # original full image
    H_orig, W_orig = img_orig.shape[:2]

    # Attempt warp (we keep original untouched)
    if settings["do_warp"]:
        warped, M, Minv = perspective_warp(img_orig)
    else:
        warped = img_orig.copy()
        M = np.eye(3, dtype=np.float32)
        Minv = np.eye(3, dtype=np.float32)

    # Work on warped copy for grid detection
    warped_gray = normalize_el(warped,
                               clahe_clip=settings["clahe_clip"],
                               tile=settings["clahe_tile"],
                               blur_ksize=settings["blur_ksize"])

    # Optionally deskew (affects warped image only)
    if settings["do_deskew"]:
        warped = auto_deskew(warped, warped_gray)
        warped_gray = normalize_el(warped,
                                   clahe_clip=settings["clahe_clip"],
                                   tile=settings["clahe_tile"],
                                   blur_ksize=settings["blur_ksize"])

    # Auto detection or manual
    if not settings["use_manual"]:
        vert_map, horiz_map = detect_grid_lines(warped_gray,
                                                polarity=settings["polarity"],
                                                binarize=settings["binarize"],
                                                ksize_v=settings["ksize_v"],
                                                ksize_h=settings["ksize_h"])
        cells_warp = grid_cells_from_maps(warped, vert_map, horiz_map,
                                          min_cell_w=settings["min_cell_w"],
                                          min_cell_h=settings["min_cell_h"])
    else:
        cells_warp = []
        # evenly split warped image
        h_w, w_w = warped.shape[:2]
        cell_w = (w_w - settings["manual_margin"]*2) // settings["n_cols"]
        cell_h = (h_w - settings["manual_margin"]*2) // settings["n_rows"]
        for r in range(settings["n_rows"]):
            for c in range(settings["n_cols"]):
                x0 = settings["manual_margin"] + c * cell_w
                y0 = settings["manual_margin"] + r * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                crop = warped[y0:y1, x0:x1].copy()
                cells_warp.append({
                    "row": r,
                    "col": c,
                    "bbox_warp": (x0, y0, x1, y1),
                    "image_warp": crop
                })

    # Build masks from warped gray & map bboxes to original coords
    outputs = []
    warped_gray_for_masks = normalize_el(warped, clahe_clip=settings["clahe_clip"], tile=settings["clahe_tile"], blur_ksize=1)
    for cell in cells_warp:
        bbox_w = cell["bbox_warp"]
        mask_w = build_mask_for_bbox(warped_gray_for_masks, bbox_w)
        # Map warped bbox back to original image coords
        bbox_orig = warp_bbox_to_original(bbox_w, Minv, clip_shape=(H_orig, W_orig))
        x_o, y_o, w_o, h_o = bbox_orig
        orig_crop = None
        if w_o > 0 and h_o > 0:
            orig_crop = img_orig[y_o:y_o+h_o, x_o:x_o+w_o].copy()
        outputs.append({
            "row": cell["row"],
            "col": cell["col"],
            "bbox_warp": bbox_w,
            "bbox_orig": bbox_orig,
            "image_warp": cell["image_warp"],
            "image_orig": orig_crop,
            "mask_warp": mask_w
        })

    # Visualization overlay (on warped plane)
    overlay_warp = warped.copy()
    for out in outputs:
        x0, y0, x1, y1 = out["bbox_warp"]
        cv2.rectangle(overlay_warp, (x0, y0), (x1, y1), (0,255,0), 2)

    # Prepare ZIP in memory with images and metadata
    files: Dict[str, bytes] = {}
    summary = {"n_cells": len(outputs), "cells": []}

    # Optional save to disk for debugging
    if save_root is not None:
        save_root = Path(save_root)
        ensure_dir(save_root)
        save_image(save_root / "overlay_warp.jpg", overlay_warp)

    for i, out in enumerate(outputs):
        r = out["row"]
        c = out["col"]
        # warp-space crop
        if out["image_warp"] is not None and out["image_warp"].size != 0:
            pil_warp = cv_to_pil(out["image_warp"])
            buf = io.BytesIO()
            pil_warp.save(buf, format="PNG")
            files[f"cell_{r:02d}_{c:02d}_warp.png"] = buf.getvalue()
            if save_root is not None:
                save_image(save_root / f"cell_{r:02d}_{c:02d}_warp.jpg", out["image_warp"])
        # original-space crop
        if out["image_orig"] is not None:
            pil_orig = cv_to_pil(out["image_orig"])
            buf = io.BytesIO()
            pil_orig.save(buf, format="PNG")
            files[f"cell_{r:02d}_{c:02d}_orig.png"] = buf.getvalue()
            if save_root is not None:
                save_image(save_root / f"cell_{r:02d}_{c:02d}_orig.jpg", out["image_orig"])
        # mask (warp-space)
        if out["mask_warp"] is not None and out["mask_warp"].size != 0:
            pil_mask = Image.fromarray((out["mask_warp"]*255).astype(np.uint8))
            buf = io.BytesIO()
            pil_mask.save(buf, format="PNG")
            files[f"cell_{r:02d}_{c:02d}_mask.png"] = buf.getvalue()
            if save_root is not None:
                mask_vis = (out["mask_warp"]*255).astype(np.uint8)
                save_image(save_root / f"cell_{r:02d}_{c:02d}_mask.jpg", cv2.cvtColor(cv2.merge([mask_vis]*3), cv2.COLOR_RGB2BGR))
        summary["cells"].append({
            "row": r,
            "col": c,
            "bbox_warp": out["bbox_warp"],
            "bbox_orig": out["bbox_orig"]
        })

    # overlay warp image
    buf = io.BytesIO()
    cv_to_pil(overlay_warp).save(buf, format="PNG")
    files["overlay_warp.png"] = buf.getvalue()
    # save summary json
    files["summary.json"] = json.dumps(summary, indent=2).encode("utf-8")

    zip_bytes = zip_bytes_from_dict(files)

    elapsed = time.time() - t0
    result = {
        "n_cells": len(outputs),
        "overlay_warp": overlay_warp,
        "outputs": outputs,
        "zip_bytes": zip_bytes,
        "summary": summary,
        "elapsed": elapsed
    }
    return result

# Run pipeline
if start_btn:
    out_base = Path(out_dir_str) if out_dir_str else None
    if not uploads:
        st.warning("Please upload at least one image.")
    else:
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            save_dir = (out_base / Path(upl.name).stem) if out_base else None
            res = process_single_image(
                img_pil,
                settings={
                    "clahe_clip": clahe_clip,
                    "clahe_tile": clahe_tile,
                    "blur_ksize": blur_ksize,
                    "do_warp": do_warp,
                    "do_deskew": do_deskew,
                    "polarity": polarity,
                    "binarize": binarize,
                    "ksize_v": ksize_v,
                    "ksize_h": ksize_h,
                    "min_cell_w": min_cell_w,
                    "min_cell_h": min_cell_h,
                    "use_manual": use_manual,
                    "n_rows": int(n_rows),
                    "n_cols": int(n_cols),
                    "manual_margin": int(manual_margin)
                },
                save_root=save_dir
            )
            st.success(f"Processed {upl.name}: {res['n_cells']} cells in {res['elapsed']:.2f}s")
            st.image(cv_to_pil(res["overlay_warp"]), caption=f"Grid overlay (warped plane): {upl.name}", use_column_width=True)

            # Show a selection of cropped cells (orig-space if available)
            to_show = res["outputs"][:min(12, len(res["outputs"]))]
            cols_show = st.columns(min(6, max(1, 6)))
            for i, out in enumerate(to_show):
                img_to_show = out["image_orig"] if out["image_orig"] is not None else out["image_warp"]
                if img_to_show is None or img_to_show.size == 0:
                    continue
                cols_show[i % len(cols_show)].image(cv_to_pil(img_to_show), caption=f"r{out['row']} c{out['col']}", use_column_width=True)

            # Provide in-memory ZIP download
            zip_name = f"{Path(upl.name).stem}_cells.zip"
            st.download_button(f"📦 Download crops for {upl.name}", data=res["zip_bytes"], file_name=zip_name, mime="application/zip")

st.markdown("---")
st.caption("Notes: This version maps detected grid cells from the warped (rectified) module plane back to the original image coordinates and exports both warp-space and original-space crops + masks. If detection fails, try manual rows×cols fallback or adjust kernel sizes / thresholds.")
