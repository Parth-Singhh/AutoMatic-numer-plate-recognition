"""
anpr.py — Robust Automatic Number Plate Recognition Module
===========================================================
Multi-strategy pipeline that works across varied lighting,
angles, and image quality — not just a single tuned image.

Strategies (tried in order until one succeeds):
  1. CLAHE + Auto-Canny  (best for most real-world images)
  2. Adaptive Threshold  (handles uneven lighting / shadows)
  3. Morphological Close (fills gaps when edges are broken)

Author : Parth Singh
"""

import cv2
import numpy as np
import pytesseract
import imutils

# ── Windows Tesseract path ──────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ── Plate geometry constraints ───────────────────────────────────────────────
# Real number plates are wide rectangles.  Filtering by aspect ratio and area
# removes false positives (wheels, windows, bumpers, etc.)
PLATE_ASPECT_MIN = 1.5    # width / height  — minimum
PLATE_ASPECT_MAX = 6.0    # width / height  — maximum
PLATE_AREA_MIN   = 800    # px²  — ignore tiny rectangles
PLATE_AREA_MAX   = 80_000 # px²  — ignore huge regions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Compute Canny thresholds automatically from the median pixel intensity.
    Works far better than hard-coded values across different exposures.
    """
    median   = np.median(gray)
    lower    = int(max(0,   (1.0 - sigma) * median))
    upper    = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation.
    Normalises local contrast so dark / bright regions are treated equally.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _is_valid_plate_contour(approx: np.ndarray, image_area: int) -> bool:
    """
    Return True only if the 4-corner contour passes plate geometry checks.
    """
    x, y, w, h = cv2.boundingRect(approx)
    if h == 0:
        return False
    aspect = w / float(h)
    area   = w * h
    return (
        PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX
        and PLATE_AREA_MIN <= area <= PLATE_AREA_MAX
    )


def _find_plate_contour(edge_map: np.ndarray, image_area: int):
    """
    Given an edge/binary map, find the best rectangular plate candidate.
    Returns the approx contour array or None.
    """
    cnts, _ = cv2.findContours(
        edge_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    # Sort largest area first — plate is usually a prominent region
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx    = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4 and _is_valid_plate_contour(approx, image_area):
            return approx
    return None


def _preprocess_plate_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Prepare the cropped plate region for Tesseract:
      1. Upscale  — Tesseract accuracy improves significantly above ~120 px height
      2. Denoise  — mild bilateral filter
      3. Threshold — Otsu binarisation for clean black/white text
    """
    # Upscale to at least 60 px height (keeps aspect ratio)
    scale  = max(1, 60 // crop.shape[0])
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Denoise
    crop = cv2.bilateralFilter(crop, 9, 75, 75)

    # Otsu threshold
    _, crop = cv2.threshold(crop, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return crop


def _run_ocr(crop: np.ndarray) -> str:
    """
    Run Tesseract with multiple PSM modes and return the best (longest) result.
    PSM 7  = single text line   (most plates)
    PSM 8  = single word
    PSM 6  = uniform block of text
    """
    cfg_base = "--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm"
    results  = []
    for psm in (7, 8, 6):
        raw = pytesseract.image_to_string(crop, config=f"{cfg_base} {psm}").strip()
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if cleaned:
            results.append(cleaned)

    # Return the longest non-empty result (more characters = more likely correct)
    return max(results, key=len) if results else ""


# ── Main public function ──────────────────────────────────────────────────────

def detect_plate(image_bgr: np.ndarray) -> dict:
    """
    Run the full ANPR pipeline on a BGR image (as loaded by OpenCV).

    Returns a dict:
        {
          "plate_contour" : np.ndarray | None,   # 4-point contour
          "cropped_raw"   : np.ndarray | None,   # raw grayscale crop
          "cropped_ocr"   : np.ndarray | None,   # preprocessed crop sent to OCR
          "plate_text"    : str,                 # recognised text  ("" if failed)
          "method"        : str,                 # which strategy succeeded
          "annotated"     : np.ndarray,          # original image with box drawn
          "success"       : bool
        }
    """
    result = {
        "plate_contour": None,
        "cropped_raw":   None,
        "cropped_ocr":   None,
        "plate_text":    "",
        "method":        "none",
        "annotated":     image_bgr.copy(),
        "success":       False
    }

    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    # ── Base preprocessing common to all strategies ──────────────────────────
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = _apply_clahe(gray)
    blurred = cv2.bilateralFilter(gray_eq, 11, 17, 17)

    # ══════════════════════════════════════════════════════════════════════════
    # Strategy 1 — CLAHE + Auto-Canny
    # Best for well-lit images with reasonable contrast.
    # ══════════════════════════════════════════════════════════════════════════
    edged1   = _auto_canny(blurred)
    contour1 = _find_plate_contour(edged1, image_area)
    if contour1 is not None:
        result["plate_contour"] = contour1
        result["method"]        = "CLAHE + Auto-Canny"
    
    # ══════════════════════════════════════════════════════════════════════════
    # Strategy 2 — Adaptive Threshold (fallback)
    # Handles uneven lighting, shadows, or very dark/bright images.
    # ══════════════════════════════════════════════════════════════════════════
    if result["plate_contour"] is None:
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=19, C=9
        )
        contour2 = _find_plate_contour(adaptive, image_area)
        if contour2 is not None:
            result["plate_contour"] = contour2
            result["method"]        = "Adaptive Threshold"

    # ══════════════════════════════════════════════════════════════════════════
    # Strategy 3 — Morphological Close (second fallback)
    # Closes broken edges — useful for low-res or noisy images.
    # ══════════════════════════════════════════════════════════════════════════
    if result["plate_contour"] is None:
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph   = cv2.morphologyEx(
            _auto_canny(gray),        # raw gray (no CLAHE) for variety
            cv2.MORPH_CLOSE, kernel
        )
        contour3 = _find_plate_contour(morph, image_area)
        if contour3 is not None:
            result["plate_contour"] = contour3
            result["method"]        = "Morphological Close"

    # ── If no strategy found a plate, return early ───────────────────────────
    if result["plate_contour"] is None:
        return result

    # ── Crop + OCR ───────────────────────────────────────────────────────────
    x, y, w, h      = cv2.boundingRect(result["plate_contour"])
    crop_raw         = gray[y:y+h, x:x+w]
    crop_ocr         = _preprocess_plate_for_ocr(crop_raw.copy())
    plate_text       = _run_ocr(crop_ocr)

    result["cropped_raw"]  = crop_raw
    result["cropped_ocr"]  = crop_ocr
    result["plate_text"]   = plate_text
    result["success"]      = True

    # ── Annotate original image ───────────────────────────────────────────────
    annotated = image_bgr.copy()
    cv2.drawContours(annotated, [result["plate_contour"]], -1, (0, 230, 255), 2)
    # Label with detected text
    cv2.putText(
        annotated, plate_text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 230, 255), 2, cv2.LINE_AA
    )
    result["annotated"] = annotated

    return result


# ── Standalone CLI usage ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "images/1.jpg"
    img      = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Could not load image: {img_path}")
        sys.exit(1)

    img = imutils.resize(img, width=800)
    out = detect_plate(img)

    print(f"\nStrategy used : {out['method']}")
    print(f"Plate detected: {'YES' if out['success'] else 'NO'}")
    print(f"Plate text    : {out['plate_text'] or '(empty)'}")

    if out["success"]:
        cv2.imshow("Annotated",      out["annotated"])
        cv2.imshow("Plate (raw)",    out["cropped_raw"])
        cv2.imshow("Plate (OCR in)", out["cropped_ocr"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nNo plate found. Tips:")
        print("  - Make sure the plate is clearly visible and not too angled")
        print("  - Try a higher resolution image")
        print("  - Ensure good contrast between plate and vehicle body")