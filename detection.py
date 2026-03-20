"""
detection.py — Cell detection for fluorescence imaging.

Public API
----------
detect_iterative(max_proj_norm, **params)
    Multi-pass LoG with masked subtraction — finds cells that single-pass LoG
    misses because brighter neighbours suppress their LoG response.
    Returns: labeled (H,W int32), blobs (N,3) [y,x,r]

fit_ellipse_pca(points)
    Fit an ellipse to 2–N boundary points via PCA.
    Returns: cy, cx, semi_major, semi_minor, angle_deg

In all cases:
    labeled   pixel value = cell index 1…N, 0 = background
    blobs     (y, x, equivalent_radius) — centroid + radius of equal-area circle
"""

import numpy as np
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.filters import gaussian

# ── shared defaults ────────────────────────────────────────────────────────────
# These values are used by detect_iterative() and can be overridden at call-time
# via **kw, or tuned interactively through the calibration step in the pipeline.
DEFAULT = dict(
    # Standard deviation (px) of the Gaussian blur applied to the max projection
    # before LoG detection.Larger values smooth out shot noise but may merge
    gauss_sigma=6.0,

    # Smallest LoG sigma to test (px).  The pipeline overwrites this after the
    # interactive calibration step (sigma ≈ cell_radius / sqrt(2)).
    min_sigma=6,

    # Largest LoG sigma to test (px).  Also overwritten by calibration.
    max_sigma=30,

    # Number of sigma values sampled logarithmically between min_sigma and
    # max_sigma.  More steps give finer size resolution at the cost of runtime.
    num_sigma=30,

    # Absolute LoG response threshold.  Blobs with a peak LoG response below this
    # value are discarded.  Lower values detect dimmer/weaker cells but increase
    # false positives; raise if background noise is triggering spurious detections.
    blob_threshold=0.1,

    # Maximum allowed fractional overlap between two detected blobs (0–1).  Blobs
    # that overlap by more than this fraction are merged.  Lower values keep 
    #more closely packed cells separate; higher values suppress duplicate 
    #detections of the same cell.
    overlap=0.75,
)


# ── shared utility ─────────────────────────────────────────────────────────────
def fit_ellipse_pca(points):
    """
    Fit an ellipse to 2–N boundary sample points via PCA on their coordinates.

    Computes the centroid of the points, runs PCA on the centred coordinate
    matrix, and projects each point onto each principal axis to determine
    semi-axis lengths.  Works with as few as 2 points (degrades gracefully
    to a circle for collinear points).

    Parameters
    ----------
    points : list of (y, x) tuples
        Boundary sample points in image (row, col) coordinates.

    Returns
    -------
    cy         : float  Row coordinate of the ellipse centre.
    cx         : float  Column coordinate of the ellipse centre.
    semi_major : float  Semi-major axis length in pixels  (≥ semi_minor).
    semi_minor : float  Semi-minor axis length in pixels.
    angle_deg  : float  Rotation of the major axis from the x-axis, CCW, in
                        degrees — matches the matplotlib Ellipse and skimage
                        regionprops orientation convention.
    """
    pts    = np.array(points, dtype=float)   # (N, 2)  rows=y, cols=x
    cy, cx = pts.mean(axis=0)
    pts_c  = pts - np.array([cy, cx])

    if len(pts_c) < 2:
        return float(cy), float(cx), 5.0, 5.0, 0.0

    cov              = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order            = np.argsort(eigvals)[::-1]   # descending eigenvalue order
    eigvecs          = eigvecs[:, order]

    # Semi-axis = max absolute projection of any point onto that principal axis
    proj_major = np.abs(pts_c @ eigvecs[:, 0])
    proj_minor = np.abs(pts_c @ eigvecs[:, 1])
    semi_major = max(float(proj_major.max()), 1.0)
    semi_minor = max(float(proj_minor.max()), 1.0)

    # Angle: major eigenvector is [dy, dx]; angle from x-axis, CCW
    major_dir = eigvecs[:, 0]
    angle_deg = float(np.degrees(np.arctan2(major_dir[0], major_dir[1])))

    return float(cy), float(cx), semi_major, semi_minor, angle_deg


# ── internal helper ────────────────────────────────────────────────────────────
def _run_log(proj_blur, min_sigma, max_sigma, num_sigma, blob_threshold, overlap):
    """
    Run LoG blob detection and return an (N, 3) array of (y, x, radius).

    skimage blob_log returns (y, x, sigma); radius = sigma * sqrt(2).
    """
    blobs = blob_log(
        proj_blur,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=blob_threshold,
        overlap=overlap,
    )
    if len(blobs) == 0:
        return np.zeros((0, 3))
    blobs[:, 2] *= np.sqrt(2)   # sigma → radius
    return blobs


# ── iterative LoG with masked subtraction ─────────────────────────────────────
def detect_iterative(max_proj_norm, max_passes=10, mask_radius_factor=0.15, **kw):
    """
    Multi-pass LoG detection with masked subtraction.

    Motivation
    ----------
    Single-pass LoG misses cells that are adjacent to brighter neighbours
    because the brighter cell's LoG response raises the local baseline,
    pushing the weaker cell's response below threshold.  By masking out
    already-detected cells and re-running LoG on the residual, those
    previously-shadowed cells become the local maxima and are found.

    Algorithm
    ---------
    Pass 1 : Gaussian-blur the max projection; run LoG → detect N₁ cells.
    Pass k : Replace each detected cell's pixels (tight disk = mask_radius_factor
             × detected radius) with their local background value (5th percentile
             of that cell's pixel intensities).  Run LoG on the modified image →
             detect Nₖ additional cells whose centroids do not overlap any
             previously detected cell.
    Stop   : when a pass finds 0 new cells, or max_passes is reached.

    The final labeled mask is built from all detected blobs using full-radius
    disks so the masks cover the complete cell area.

    Parameters
    ----------
    max_proj_norm      : np.ndarray (H, W) float64 in [0, 1]
        Normalised temporal max projection.
    max_passes         : int   Maximum number of detection passes (default 10).
    mask_radius_factor : float Fraction of detected radius used for the
                               subtraction mask (default 0.15).  Values < 1.0
                               prevent the mask from bleeding into adjacent cells.
    **kw               : optional overrides for DEFAULT parameters.

    Returns
    -------
    labeled : np.int32 (H, W)   Pixel value = cell index (1…N), 0 = background.
    blobs   : np.ndarray (N, 3) [y, x, r]  All cells found across all passes.
    """
    p = {**DEFAULT, **kw}
    H, W = max_proj_norm.shape

    # Gaussian blur once — reused across all passes
    proj_blur = gaussian(max_proj_norm, sigma=p["gauss_sigma"])
    working   = proj_blur.copy()

    all_blobs    = []                                    # accumulates [y, x, r] across passes
    claimed_mask = np.zeros((H, W), dtype=bool)          # centroids already detected

    for pass_num in range(1, max_passes + 1):
        raw_blobs = _run_log(working, p["min_sigma"], p["max_sigma"],
                             p["num_sigma"], p["blob_threshold"], p["overlap"])

        if len(raw_blobs) == 0:
            print(f"  [iterative] Pass {pass_num}: no blobs found — stopping")
            break

        # Keep only blobs whose centroid pixel is not already claimed
        new_blobs = []
        for (y, x, r) in raw_blobs:
            yi = int(np.clip(round(y), 0, H - 1))
            xi = int(np.clip(round(x), 0, W - 1))
            if not claimed_mask[yi, xi]:
                new_blobs.append([y, x, r])

        print(f"  [iterative] Pass {pass_num}: "
              f"{len(new_blobs)} new cells  "
              f"({len(raw_blobs) - len(new_blobs)} already-detected re-detections skipped)")

        if len(new_blobs) == 0:
            print(f"  [iterative] Pass {pass_num}: no genuinely new cells — stopping")
            break

        for (y, x, r) in new_blobs:
            all_blobs.append([y, x, r])

            # Mark centroid as claimed
            yi = int(np.clip(round(y), 0, H - 1))
            xi = int(np.clip(round(x), 0, W - 1))
            claimed_mask[yi, xi] = True

            # Tight mask for subtraction — undersized to avoid bleeding into neighbours
            tight_r = max(1.0, r * mask_radius_factor)
            rr, cc  = disk((y, x), tight_r, shape=(H, W))

            if len(rr) == 0:
                continue

            # Replace cell pixels with local background (5th-percentile of cell)
            bg_val               = float(np.percentile(working[rr, cc], 5))
            working[rr, cc]      = bg_val
            claimed_mask[rr, cc] = True   # prevent re-detection at these pixels too

    print(f"  [iterative] Total: {len(all_blobs)} cells across {pass_num} pass(es)")

    # Build final labeled mask using full-radius disks
    labeled = np.zeros((H, W), dtype=np.int32)
    for i, (y, x, r) in enumerate(all_blobs):
        rr, cc = disk((y, x), max(1.0, r), shape=(H, W))
        labeled[rr, cc] = i + 1

    blobs_arr = np.array(all_blobs) if all_blobs else np.zeros((0, 3))
    return labeled, blobs_arr
