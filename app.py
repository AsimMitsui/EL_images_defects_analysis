import cv2
import numpy as np
from typing import List, Tuple, Optional

def preprocess_for_threshold(img_bgr: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    # Contrast limited adaptive histogram equalization can help
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def find_module_edges(gray: np.ndarray) -> np.ndarray:
    # Try to find the largest contour (module outline) to crop to module area
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # add small padding
    pad = int(0.01 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return gray[y0:y1, x0:x1], (x0, y0)

def threshold_image(gray: np.ndarray, adaptive: bool = True, blockSize: int = 35, C: int = 5) -> np.ndarray:
    if adaptive:
        # blockSize must be odd and >=3
        blockSize = max(3, blockSize if blockSize % 2 == 1 else blockSize + 1)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize, C)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def clean_mask(th: np.ndarray, morph_kernel: int = 5) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    return closed

def extract_bboxes(mask: np.ndarray, min_area: int = 1000, max_area: Optional[int] = None,
                   aspect_min: float = 0.3, aspect_max: float = 2.0) -> Tuple[List[Tuple[int,int,int,int]], List[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    masks = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area and area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        if ar < aspect_min or ar > aspect_max:
            continue
        # create mask per bbox (cropped mask)
        bboxes.append((x, y, w, h))
        # crop corresponding mask and normalize
        m_crop = mask[y:y+h, x:x+w].copy()
        masks.append((m_crop > 0).astype(np.uint8))
    # sort loosely by y then x
    bboxes_sorted = sorted(bboxes, key=lambda b: (b[1], b[0]))
    # try to reorder into rows (grid)
    bboxes_grid = sort_bboxes_into_rows(bboxes_sorted)
    return bboxes_grid, masks

def sort_bboxes_into_rows(bboxes: List[Tuple[int,int,int,int]], row_tol: int = 25) -> List[Tuple[int,int,int,int]]:
    if not bboxes:
        return []
    rows = []
    for b in sorted(bboxes, key=lambda b: b[1]):  # sort by y
        x,y,w,h = b
        placed = False
        for row in rows:
            # row has average y; compare
            ys = [r[1] for r in row]
            avg_y = sum(ys) / len(ys)
            if abs(y - avg_y) <= row_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    # sort each row by x
    rows_sorted = [sorted(row, key=lambda r: r[0]) for row in rows]
    # flatten with row order top-to-bottom
    flattened = [b for row in rows_sorted for b in row]
    return flattened

def detect_cells_from_image(image_bgr: np.ndarray,
                            blur_ksize: int = 5,
                            adaptive: bool = True,
                            th_blocksize: int = 35,
                            th_C: int = 5,
                            morph_kernel: int = 5,
                            min_area: int = 1000,
                            max_area: Optional[int] = None,
                            aspect_min: float = 0.4,
                            aspect_max: float = 1.6,
                            row_tol: int = 25) -> dict:
    """
    Returns dict with:
      - bboxes: list of (x,y,w,h)
      - masks: list of binary masks for each bbox (aligned to bbox)
      - module_crop_offset: (x0,y0) if module cropping was used
    """
    gray = preprocess_for_threshold(image_bgr, blur_ksize=blur_ksize)
    maybe = find_module_edges(gray)
    if isinstance(maybe, tuple):
        gray_cropped, offset = maybe
    else:
        gray_cropped = maybe
        offset = (0,0)

    th = threshold_image(gray_cropped, adaptive=adaptive, blockSize=th_blocksize, C=th_C)
    clean = clean_mask(th, morph_kernel=morph_kernel)

    bboxes_rel, masks = extract_bboxes(clean, min_area=min_area, max_area=max_area,
                                       aspect_min=aspect_min, aspect_max=aspect_max)
    # Convert relative bboxes back to original coordinates
    xoff, yoff = offset
    bboxes = [(x + xoff, y + yoff, w, h) for (x,y,w,h) in bboxes_rel]

    # Recreate full masks aligned to bbox in full image coordinates
    masks_full = []
    for (x_rel, y_rel, w, h), m in zip(bboxes_rel, masks):
        masks_full.append(m)  # still bbox-local mask

    # Try to apply small heuristic to enforce grid ordering using row_tol
    bboxes = sort_bboxes_into_rows(bboxes, row_tol=row_tol)

    return {"bboxes": bboxes, "masks": masks_full, "module_crop_offset": offset, "binary_mask": clean}

def draw_bboxes_on_image(img_bgr: np.ndarray, bboxes: List[Tuple[int,int,int,int]], color=(0,255,0)) -> np.ndarray:
    out = img_bgr.copy()
    for i, (x,y,w,h) in enumerate(bboxes):
        cv2.rectangle(out, (x,y), (x+w, y+h), color, 2)
        cv2.putText(out, str(i+1), (x+4, y+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def crop_cells_from_image(img_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x,y,w,h = bbox
    h_img, w_img = img_bgr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    return img_bgr[y0:y1, x0:x1].copy()
