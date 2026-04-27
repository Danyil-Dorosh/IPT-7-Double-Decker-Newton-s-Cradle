"""
detection.py
------------
Finds silver balls in a single video frame.

Pipeline (called once per frame):
  1. build_mask()   → convert frame to HSV, keep bright+unsaturated pixels
  2. detect_balls() → try Hough circles; fall back to contour method
  3. Returns a list of (cx, cy, radius) tuples sorted left→right

You do NOT need to understand OpenCV internals to use this module.
Just call detect_balls(frame, expected_radius_px) and get ball positions back.
"""

import cv2
import numpy as np

from temp.config import (
    SILVER_MIN_V, SILVER_MAX_S,
    HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2,
    HOUGH_RMIN_FRAC, HOUGH_RMAX_FRAC, HOUGH_MINDIST_FRAC,
    CONTOUR_MIN_AREA_FRAC, CONTOUR_MAX_AREA_FRAC, CONTOUR_MIN_CIRCULARITY,
)

# Type alias for clarity
BallList = list[tuple[int, int, int]]   # [(cx, cy, radius), ...]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — BUILD MASK
# ─────────────────────────────────────────────────────────────────────────────

def build_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a black-and-white image where WHITE = potential ball pixel.

    How it works:
    - Convert from BGR (camera format) to HSV (Hue-Saturation-Value)
    - Silver balls are BRIGHT (high V) and have NO strong colour (low S)
    - Brown cardboard has medium V and medium-high S → gets filtered out
    - cv2.inRange() keeps only pixels inside the [min, max] HSV range
    - Morphological operations clean up noise:
        OPEN  = erosion then dilation → removes tiny speckles
        CLOSE = dilation then erosion → fills small holes inside blobs
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Keep: any hue (0-180), low saturation, high brightness
    lower = np.array([0,           0,           SILVER_MIN_V])
    upper = np.array([180,  SILVER_MAX_S,        255        ])
    mask  = cv2.inRange(hsv, lower, upper)

    # Clean up noise with a 5×5 ellipse-shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return mask  # shape: (H, W), values: 0 or 255


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2a — HOUGH CIRCLE DETECTOR (primary method)
# ─────────────────────────────────────────────────────────────────────────────

def _hough_circles(frame_bgr: np.ndarray,
                   mask: np.ndarray,
                   expected_r: int) -> BallList:
    """
    Uses the Hough Transform to find circles.

    The Hough Transform works by voting:
    - For each bright pixel, consider all circles that could pass through it
    - Each such circle gets one vote in an "accumulator" grid
    - Circles with many votes (= many pixels agree on them) are real circles

    We apply the mask first so it only looks at silver-coloured regions.

    Parameters tuned for Newton's cradle:
    - minDist: balls are nearly touching, so set to ~1.8× radius
    - param2: low value (25) = sensitive but may give false positives
    """
    # Apply mask to greyscale image
    gray   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_and(gray, gray, mask=mask)
    # Blur reduces noise from specular reflections on the balls
    gray   = cv2.GaussianBlur(gray, (9, 9), 2)

    r_min  = max(5, int(expected_r * HOUGH_RMIN_FRAC))
    r_max  = int(expected_r * HOUGH_RMAX_FRAC)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp       = HOUGH_DP,
        minDist  = int(expected_r * HOUGH_MINDIST_FRAC),
        param1   = HOUGH_PARAM1,
        param2   = HOUGH_PARAM2,
        minRadius= r_min,
        maxRadius= r_max,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    return [(c[0], c[1], c[2]) for c in circles]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — CONTOUR FALLBACK (when Hough fails)
# ─────────────────────────────────────────────────────────────────────────────

def _contour_circles(mask: np.ndarray, expected_r: int) -> BallList:
    """
    Alternative detector: find blobs by their outline (contour).

    A contour is just the boundary pixels of a white region in the mask.
    We filter blobs by:
    1. Area — must be roughly the size of a ball
    2. Circularity — formula: 4π·Area / Perimeter²
                     = 1.0 for a perfect circle, <1 for other shapes

    This is more robust to motion blur than Hough, but less precise.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Expected area of a circle with radius expected_r
    area_expected = np.pi * expected_r ** 2
    area_min = area_expected * CONTOUR_MIN_AREA_FRAC ** 2
    area_max = area_expected * CONTOUR_MAX_AREA_FRAC ** 2

    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (area_min < area < area_max):
            continue                          # wrong size → skip

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < CONTOUR_MIN_CIRCULARITY:
            continue                          # not round enough → skip

        # Get the smallest enclosing circle of the contour
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        balls.append((int(cx), int(cy), int(r)))

    return balls


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — call this from main
# ─────────────────────────────────────────────────────────────────────────────

def detect_balls(frame_bgr: np.ndarray, expected_r: int) -> BallList:
    """
    Main entry point. Returns up to 5 (cx, cy, radius) tuples, sorted left→right.

    Strategy:
    - Always build the colour mask first
    - Try Hough circles (more precise)
    - If fewer than 3 detected, fall back to contour method (more robust)
    - Keep best 5 by closeness to expected radius
    - Sort left→right by x coordinate
    """
    mask    = build_mask(frame_bgr)
    circles = _hough_circles(frame_bgr, mask, expected_r)

    if len(circles) < 3:
        circles = _contour_circles(mask, expected_r)

    if not circles:
        return []

    # Keep up to 5 detections closest in size to the expected ball radius
    circles = sorted(circles, key=lambda c: abs(c[2] - expected_r))[:5]
    # Sort by x-position: ball 1 is leftmost, ball 5 is rightmost
    circles = sorted(circles, key=lambda c: c[0])
    return circles


def estimate_radius_from_frames(cap: cv2.VideoCapture,
                                rough_r: int,
                                n_frames: int = 30) -> list:
    """
    Read first n_frames frames and return the first detection with ≥4 balls.
    Used for initial calibration before the main loop.
    """
    found = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        circles = detect_balls(frame, rough_r)
        if len(circles) >= 4:
            found = circles
            break
    return found
