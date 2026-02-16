import os
import sys
import json
import argparse
from dataclasses import dataclass

import cv2
import numpy as np

# -----------------------------
# Making SAM2 importable if repo is ./sam2
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_REPO_DIR = os.path.join(PROJECT_DIR, "sam2")
if os.path.exists(SAM2_REPO_DIR) and SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def imread_color(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def to_u8_mask(mask) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)

def segmented_image(bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(bgr, bgr, mask=mask_u8)

def alpha_overlay(bgr: np.ndarray, mask_u8: np.ndarray, color_bgr=(0, 0, 255), alpha=0.45) -> np.ndarray:
    out = bgr.copy()
    m = (mask_u8 > 0)
    if m.ndim == 3:
        m = m[:, :, 0]
    color_layer = np.zeros_like(bgr, dtype=np.uint8)
    color_layer[:] = color_bgr
    blended = cv2.addWeighted(bgr, 1 - alpha, color_layer, alpha, 0)
    out[m] = blended[m]
    return out

def draw_contour_overlay(bgr: np.ndarray, mask_u8: np.ndarray, contour_color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = bgr.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, contour_color, thickness)
    return out

def largest_component(mask_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_u8
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == best] = 255
    return out

def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes using flood fill from borders."""
    h, w = mask_u8.shape[:2]
    inv = cv2.bitwise_not(mask_u8)
    flood = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)  # remove background connected to border
    holes = flood  # remaining white are holes in inv => holes in mask
    holes = cv2.bitwise_not(holes)
    return cv2.bitwise_or(mask_u8, holes)

def mask_to_bbox_xyxy(mask_u8: np.ndarray, pad: int = 10):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask_u8.shape[1] - 1, x1 + pad)
    y1 = min(mask_u8.shape[0] - 1, y1 + pad)
    return (x0, y0, x1, y1)

def save_outputs(outdir: str, prefix: str, bgr: np.ndarray, mask_u8: np.ndarray,
                 overlay_color=(0, 0, 255), contour_color=(0, 255, 0)):
    ensure_dir(outdir)
    mask_u8 = to_u8_mask(mask_u8)
    ov = alpha_overlay(bgr, mask_u8, color_bgr=overlay_color, alpha=0.45)
    ov = draw_contour_overlay(ov, mask_u8, contour_color=contour_color, thickness=2)
    seg = segmented_image(bgr, mask_u8)
    cv2.imwrite(os.path.join(outdir, f"{prefix}_mask.png"), mask_u8)
    cv2.imwrite(os.path.join(outdir, f"{prefix}_overlay.png"), ov)
    cv2.imwrite(os.path.join(outdir, f"{prefix}_segmented.png"), seg)

# -----------------------------
# Classical thermal segmentation (OpenCV only)
# -----------------------------
@dataclass
class ClassicalConfig:
    mode: str = "hsv_v"        # gray | hsv_v | lab_l | red
    clahe: bool = True
    bilateral: bool = True
    thresh: str = "percentile" # otsu | percentile
    percentile: float = 70.0   # good start for pseudo-colored thermal
    morph: int = 9
    grabcut_iters: int = 5

def read_as_intensity(bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hsv_v":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    if mode == "lab_l":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    if mode == "red":
        return bgr[:, :, 2]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def grabcut_refine(bgr: np.ndarray, init_mask_u8: np.ndarray, iters: int) -> np.ndarray:
    """
    Robust GrabCut refinement. If initialization is degenerate (no FG or no BG samples),
    it gracefully falls back to the input mask instead of crashing.
    """
    h, w = init_mask_u8.shape[:2]
    init_mask_u8 = ((init_mask_u8 > 0).astype(np.uint8) * 255)

    fg_pixels = int((init_mask_u8 > 0).sum())
    total = h * w
    if fg_pixels == 0:
        return init_mask_u8  # no foreground
    fg_frac = fg_pixels / float(total)

    # If mask is too small or too big, GrabCut often fails; just return mask
    if fg_frac < 0.001 or fg_frac > 0.99:
        return init_mask_u8

    # Build gc_mask: start as probable BG
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Set probable FG where init mask is
    gc_mask[init_mask_u8 > 0] = cv2.GC_PR_FGD

    # sure FG = erode
    k = max(3, (int(round(min(h, w) * 0.015)) | 1))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(init_mask_u8, ker, iterations=1)
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    # sure BG: use image border + outside a modest dilation
    dil = cv2.dilate(init_mask_u8, ker, iterations=2)

    border = np.zeros((h, w), np.uint8)
    b = max(5, int(round(min(h, w) * 0.02)))
    border[:b, :] = 255
    border[-b:, :] = 255
    border[:, :b] = 255
    border[:, -b:] = 255

    sure_bg = cv2.bitwise_or(border, cv2.bitwise_not(dil))
    gc_mask[sure_bg > 0] = cv2.GC_BGD

    # Final sanity: ensure we still have both BG and FG labels
    has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
    has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
    if not (has_fg and has_bg):
        return init_mask_u8

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, None, bgdModel, fgdModel, iters, mode=cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Fallback: return init mask if GrabCut fails
        return init_mask_u8

    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def classical_segment(bgr: np.ndarray, cfg: ClassicalConfig):
    # 1. Denoise to prevent background 'speckles' from expanding
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Precise Color Masking for "Rainbow" Thermal Palettes
    # We target the 'hot' colors (Yellow/Orange/Red)
    # Range 1: Yellow to Orange
    lower_warm = np.array([10, 100, 100])
    upper_warm = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_warm, upper_warm)
    
    # Range 2: Red (Red wraps around 0 and 180 in OpenCV HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Range 3: High-Intensity White (The core heat)
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([180, 50, 255])
    mask3 = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine only these specific heat signatures
    combined = cv2.bitwise_or(mask1, mask2)
    combined = cv2.bitwise_or(combined, mask3)
    
    # 3. Clean up: Small opening to remove noise, then closing to join the dog
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Use a moderate close—too big and you'll merge with the car
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    # 4. Grab the dog (the largest central component)
    raw = largest_component(closed)
    
    # 5. GrabCut Refinement (Crucial for the "Exact Boundary")
    refined = grabcut_refine(bgr, raw, cfg.grabcut_iters)
    
    return refined, combined, raw

import numpy as np
# -----------------------------
# COmpute Metrics
# -----------------------------

def compute_metrics(mask_classical, mask_sam2):
    """
    Computes IoU and F1 Score between two binary masks.
    Assumes masks are uint8 with values 0 and 255.
    """
    # Convert to boolean for logical operations
    m1 = mask_classical > 0
    m2 = mask_sam2 > 0

    # Intersection and Union
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    # IoU calculation
    iou = intersection / union if union > 0 else 0.0

    # F1 Score / Dice Coefficient
    # F1 = (2 * TP) / (2 * TP + FP + FN) 
    # Which simplifies to (2 * intersection) / (pixels_in_m1 + pixels_in_m2)
    sum_pixels = m1.sum() + m2.sum()
    f1 = (2.0 * intersection) / sum_pixels if sum_pixels > 0 else 0.0

    return iou, f1

# -----------------------------
# SAM2 (comparison only)
# -----------------------------
def init_sam2_predictor(cfg_path: str, ckpt_path: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(cfg_path, ckpt_path, device=device)
    return SAM2ImagePredictor(model)

def sam2_predict_mask(predictor, bgr: np.ndarray, box_xyxy):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    box = np.array(box_xyxy, dtype=np.float32)[None, :]
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    if masks is None or len(masks) == 0:
        return None
    best = int(np.argmax(scores))
    return (masks[best] > 0).astype(np.uint8) * 255

# -----------------------------
# Metrics + Panel
# -----------------------------
def iou(mask_a_u8: np.ndarray, mask_b_u8: np.ndarray) -> float:
    A = (mask_a_u8 > 0)
    B = (mask_b_u8 > 0)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def boundary_f1(mask_a_u8: np.ndarray, mask_b_u8: np.ndarray, tol_px: int = 2) -> float:
    a_edge = cv2.Canny(mask_a_u8, 50, 150)
    b_edge = cv2.Canny(mask_b_u8, 50, 150)
    k = max(1, tol_px)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    a_d = cv2.dilate(a_edge, ker, iterations=1)
    b_d = cv2.dilate(b_edge, ker, iterations=1)
    a_total = (a_edge > 0).sum()
    b_total = (b_edge > 0).sum()
    if a_total == 0 or b_total == 0:
        return 0.0
    prec = ((a_edge > 0) & (b_d > 0)).sum() / a_total
    rec  = ((b_edge > 0) & (a_d > 0)).sum() / b_total
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def make_panel(bgr: np.ndarray, class_mask_u8: np.ndarray, sam_mask_u8: np.ndarray) -> np.ndarray:
    class_ov = alpha_overlay(bgr, class_mask_u8, (0, 0, 255), 0.45)
    class_ov = draw_contour_overlay(class_ov, class_mask_u8, (0, 255, 0), 2)

    sam_ov = alpha_overlay(bgr, sam_mask_u8, (255, 0, 0), 0.45)
    sam_ov = draw_contour_overlay(sam_ov, sam_mask_u8, (0, 255, 255), 2)

    disag = np.zeros_like(bgr)
    A = (class_mask_u8 > 0)
    B = (sam_mask_u8 > 0)
    disag[A & ~B] = (0, 0, 255)
    disag[~A & B] = (255, 0, 0)
    disag[A & B]  = (255, 255, 255)

    return np.concatenate([bgr, class_ov, sam_ov, disag], axis=1)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=["Thermal.jpg", "thermal2.jpg","test.jpg"])
    ap.add_argument("--outdir", default="out")

    # Classical knobs
    ap.add_argument("--mode", default="hsv_v", choices=["gray", "hsv_v", "lab_l", "red"])
    ap.add_argument("--no_clahe", action="store_true")
    ap.add_argument("--no_bilateral", action="store_true")
    ap.add_argument("--thresh", default="percentile", choices=["otsu", "percentile"])
    ap.add_argument("--percentile", type=float, default=70.0)
    ap.add_argument("--morph", type=int, default=9)
    ap.add_argument("--grabcut_iters", type=int, default=5)

    # SAM2
    ap.add_argument("--run_sam2", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sam2_checkpoint", default="")
    ap.add_argument("--sam2_cfg", default="")

    # Metrics
    ap.add_argument("--boundary_tol", type=int, default=2)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    cfg = ClassicalConfig(
        mode=args.mode,
        clahe=not args.no_clahe,
        bilateral=not args.no_bilateral,
        thresh=args.thresh,
        percentile=args.percentile,
        morph=args.morph,
        grabcut_iters=args.grabcut_iters
    )

    predictor = None
    if args.run_sam2:
        if not args.sam2_cfg or not os.path.exists(args.sam2_cfg):
            raise FileNotFoundError("Pass a valid --sam2_cfg .yaml (full path).")
        if not args.sam2_checkpoint or not os.path.exists(args.sam2_checkpoint):
            raise FileNotFoundError("Pass a valid --sam2_checkpoint .pt (full path).")
        predictor = init_sam2_predictor(args.sam2_cfg, args.sam2_checkpoint, args.device)

    for img_path in args.images:
        bgr = imread_color(img_path)
        out_sub = os.path.join(args.outdir, stem(img_path))
        ensure_dir(out_sub)

        c_mask, intensity_dbg, raw_dbg = classical_segment(bgr, cfg)
        c_mask = to_u8_mask(c_mask)

        # Debug saves
        cv2.imwrite(os.path.join(out_sub, "debug_intensity.png"), intensity_dbg)
        cv2.imwrite(os.path.join(out_sub, "debug_mask_raw.png"), raw_dbg)
        

        # Required classical outputs
        save_outputs(out_sub, "classical", bgr, c_mask, overlay_color=(0, 0, 255), contour_color=(0, 255, 0))

        metrics = {"image": img_path}
        

        # SAM2 + compare
        if predictor is not None:
            box = mask_to_bbox_xyxy(c_mask, pad=15)
            if box is None:
                print(f"[WARN] empty classical mask for {img_path}, skipping SAM2")
            else:
                s_mask = sam2_predict_mask(predictor, bgr, box)
                if s_mask is None:
                    print(f"[WARN] SAM2 produced no mask for {img_path}")
                else:
                    s_mask = to_u8_mask(s_mask)
                    save_outputs(out_sub, "sam2", bgr, s_mask, overlay_color=(255, 0, 0), contour_color=(0, 255, 255))

                    metrics["iou_classical_vs_sam2"] = iou(c_mask, s_mask)
                    metrics["boundary_f1_tol_px"] = args.boundary_tol
                    metrics["boundary_f1_classical_vs_sam2"] = boundary_f1(c_mask, s_mask, tol_px=args.boundary_tol)

                    panel = make_panel(bgr, c_mask, s_mask)
                    cv2.imwrite(os.path.join(out_sub, "comparison_panel.png"), panel)

        with open(os.path.join(out_sub, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        iou_val, f1_val = compute_metrics(c_mask, s_mask)
        metrics["iou"] = iou_val
        metrics["f1_score"] = f1_val
        print(f"Comparison: IoU={iou_val:.4f}, F1={f1_val:.4f}")

        print(f"[OK] {img_path} -> {out_sub}")

if __name__ == "__main__":
    main()
