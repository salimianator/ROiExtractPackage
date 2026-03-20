#!/usr/bin/env python3
"""
SigProcessingPipeline.py — Fluorescence cell imaging analysis pipeline.

Usage
-----
    python SigProcessingPipeline.py path/to/imaging_file.tif [--outdir path/to/output]

Steps
-----
1. Load TIFF → compute temporal max projection
2. Interactive PCA calibration → set LoG sigma bounds
3. Iterative LoG detection → labeled cell mask
4. Generate per-cell binary masks
5. Extract fluorescence traces (mean intensity per cell per frame)
6. Render overlay video with cell outlines
7. Save validation summary figure
"""

import argparse
import os
import sys

# ── portable import of detection module (same directory as this script) ────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import imageio
import matplotlib
matplotlib.use("MacOSX")   # interactive backend — required for calibration ginput
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tifffile

from skimage.draw import disk

from detection import detect_iterative, fit_ellipse_pca, DEFAULT


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fluorescence cell imaging analysis pipeline."
    )
    parser.add_argument("tiff", help="Path to the input TIFF imaging file.")
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Directory for all output files.  "
            "Defaults to the same directory as the input TIFF."
        ),
    )
    return parser.parse_args()


# ── local mutable copy of detection defaults ──────────────────────────────────
# calibrate_cell_sizes() updates min_sigma / max_sigma before detection runs
PARAMS = dict(DEFAULT)


# ── interactive calibration ───────────────────────────────────────────────────
def calibrate_cell_sizes(max_proj_norm):
    """
    Interactive warm-start: collect 4 boundary clicks on the largest visible
    cell, then 4 on the smallest, and derive LoG sigma bounds from PCA ellipses
    fitted to each set of clicks.

    The relationship between cell radius and LoG sigma is:
        r = sigma * sqrt(2)  →  sigma = r / sqrt(2)
    where r = sqrt(semi_major * semi_minor) (geometric mean of PCA semi-axes).

    Parameters
    ----------
    max_proj_norm : np.ndarray (H, W) float64 in [0, 1]
        Normalised temporal max projection displayed during calibration.

    Returns
    -------
    min_sigma : float   LoG sigma for the smallest clicked cell.
    max_sigma : float   LoG sigma for the largest clicked cell.

    Raises
    ------
    RuntimeError
        If fewer than 4 clicks are collected for either cell.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.imshow(max_proj_norm, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        "CALIBRATION  —  Step 1 of 2\n"
        "Click 4 boundary points on the LARGEST cell visible\n"
        "(e.g. top, bottom, left-edge, right-edge of that cell)",
        fontsize=11, color="white",
    )
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.05)

    # ── large cell ────────────────────────────────────────────────────────────
    print("\n[CALIBRATION] Click 4 boundary points on the LARGEST cell …")
    large_xy = plt.ginput(4, timeout=0)   # returns [(x, y), …] — matplotlib convention
    if len(large_xy) < 4:
        plt.close(fig)
        raise RuntimeError(
            f"[CALIBRATION] Expected 4 clicks for largest cell, got {len(large_xy)}. "
            "Aborting pipeline."
        )

    for (x, y) in large_xy:
        ax.plot(x, y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=0.8)

    # fit_ellipse_pca uses (y, x) — flip from ginput's (x, y)
    large_pts_yx = [(y, x) for (x, y) in large_xy]
    cy_l, cx_l, sm_l, sn_l, ang_l = fit_ellipse_pca(large_pts_yx)
    r_large = np.sqrt(sm_l * sn_l)
    ax.add_patch(mpatches.Ellipse(
        (cx_l, cy_l), 2 * sm_l, 2 * sn_l, angle=ang_l,
        fill=False, edgecolor="red", linewidth=1.5,
    ))

    # ── small cell ────────────────────────────────────────────────────────────
    ax.set_title(
        "CALIBRATION  —  Step 2 of 2\n"
        "Click 4 boundary points on the SMALLEST cell visible\n"
        "(e.g. top, bottom, left-edge, right-edge of that cell)",
        fontsize=11, color="white",
    )
    fig.canvas.draw()
    plt.pause(0.05)

    print("[CALIBRATION] Click 4 boundary points on the SMALLEST cell …")
    small_xy = plt.ginput(4, timeout=0)
    if len(small_xy) < 4:
        plt.close(fig)
        raise RuntimeError(
            f"[CALIBRATION] Expected 4 clicks for smallest cell, got {len(small_xy)}. "
            "Aborting pipeline."
        )

    for (x, y) in small_xy:
        ax.plot(x, y, "c+", markersize=10, markeredgewidth=2)

    small_pts_yx = [(y, x) for (x, y) in small_xy]
    cy_s, cx_s, sm_s, sn_s, ang_s = fit_ellipse_pca(small_pts_yx)
    r_small = np.sqrt(sm_s * sn_s)
    ax.add_patch(mpatches.Ellipse(
        (cx_s, cy_s), 2 * sm_s, 2 * sn_s, angle=ang_s,
        fill=False, edgecolor="cyan", linewidth=1.5,
    ))

    # sigma = r / sqrt(2)  (inverse of _run_log's  r = sigma * sqrt(2))
    min_sigma = max(1.0, r_small / np.sqrt(2))
    max_sigma = max(min_sigma + 1.0, r_large / np.sqrt(2))

    ax.set_title(
        f"Calibration complete\n"
        f"Large cell  r={r_large:.1f} px → max_sigma={max_sigma:.1f}  |  "
        f"Small cell  r={r_small:.1f} px → min_sigma={min_sigma:.1f}\n"
        "Close this window to continue …",
        fontsize=11, color="white",
    )
    fig.canvas.draw()
    plt.pause(0.1)
    print(f"[CALIBRATION] r_large={r_large:.1f}  r_small={r_small:.1f}")
    print(f"[CALIBRATION] min_sigma={min_sigma:.2f}  max_sigma={max_sigma:.2f}")
    input("[CALIBRATION] Review the calibration window, then press Enter here to continue …")
    plt.close(fig)

    return min_sigma, max_sigma


# ── diagnostic helper ──────────────────────────────────────────────────────────
def _save_diagnostic(label_img, max_proj_norm, title, out_path):
    """
    Save a two-panel diagnostic figure showing actual pixel masks.

    Left panel  : each cell region filled with a distinct colour, overlaid on
                  the max projection (alpha blend).
    Right panel : green boundary contours drawn on the max projection, with
                  each cell numbered at its centroid.
    """
    from skimage.color import label2rgb
    from skimage.measure import regionprops as _rp
    from skimage.segmentation import find_boundaries

    N_cells_d = int((np.unique(label_img) > 0).sum())
    n_fg_px   = int((label_img > 0).sum())

    overlay = label2rgb(label_img, image=max_proj_norm,
                        bg_label=0, alpha=0.35, kind="overlay")

    bounds   = find_boundaries(label_img, mode="thick")
    bnd_rgba = np.zeros((*max_proj_norm.shape, 4), dtype=np.float32)
    bnd_rgba[bounds] = [0.0, 1.0, 0.0, 0.9]

    fig_d, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 9))

    ax_l.imshow(overlay)
    ax_l.set_title(f"Coloured region fill  ({N_cells_d} cells)", fontsize=11)
    ax_l.axis("off")

    ax_r.imshow(max_proj_norm, cmap="gray", vmin=0, vmax=1)
    ax_r.imshow(bnd_rgba)
    for i, region in enumerate(_rp(label_img)):
        cy, cx = region.centroid
        ax_r.text(cx, cy, str(i + 1), color="yellow", fontsize=4,
                  ha="center", va="center", fontweight="bold")
    ax_r.set_title(f"Actual boundary contours  ({N_cells_d} cells)", fontsize=11)
    ax_r.axis("off")

    fig_d.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_d.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"  [DIAGNOSTIC] {N_cells_d} cells | {n_fg_px} fg px → saved {out_path}")


# ── main pipeline ──────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    tiff_path = os.path.abspath(args.tiff)
    if not os.path.isfile(tiff_path):
        sys.exit(f"ERROR: TIFF file not found: {tiff_path}")

    BASE = args.outdir if args.outdir else os.path.dirname(tiff_path)
    os.makedirs(BASE, exist_ok=True)

    def out(fname):
        return os.path.join(BASE, fname)

    # ── STEP 1: Load TIFF and build max projection ─────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading TIFF and computing max projection")
    print("=" * 60)

    with tifffile.TiffFile(tiff_path) as tif:
        data = tif.asarray()

    print(f"Raw array shape: {data.shape}, dtype: {data.dtype}")

    if data.ndim == 2:
        frames = data[np.newaxis, ...]
    elif data.ndim == 3:
        frames = data
    elif data.ndim == 4:
        if data.shape[1] in (1, 2, 3, 4) and data.shape[1] < data.shape[2]:
            frames = data[:, 0, :, :]
        elif data.shape[3] in (1, 2, 3, 4) and data.shape[3] < data.shape[1]:
            frames = data[:, :, :, 0]
        else:
            frames = data[:, 0, :, :]
    else:
        frames = data.reshape(-1, data.shape[-2], data.shape[-1])

    print(f"Frames array shape: {frames.shape}  (N_frames, H, W)")

    N_frames, H, W = frames.shape

    max_proj      = frames.max(axis=0).astype(np.float64)
    proj_min      = max_proj.min()
    proj_max      = max_proj.max()
    max_proj_norm = (max_proj - proj_min) / (proj_max - proj_min + 1e-12)

    print(f"Max projection range: [{max_proj_norm.min():.4f}, {max_proj_norm.max():.4f}]")

    # ── STEP 2: Calibration + iterative detection ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Calibration + Detection  (method = iterative)")
    print("=" * 60)

    min_sigma, max_sigma = calibrate_cell_sizes(max_proj_norm)
    PARAMS["min_sigma"] = min_sigma
    PARAMS["max_sigma"] = max_sigma
    print(f"  Calibrated sigma range: [{min_sigma:.2f}, {max_sigma:.2f}]")

    labeled_mask, blobs = detect_iterative(max_proj_norm, **PARAMS)
    N_cells = len(blobs)
    print(f"  Detected: {N_cells} cells")

    # Diagnostic: actual pixel masks after detection
    _save_diagnostic(labeled_mask, max_proj_norm,
                     "After iterative LoG detection", out("diag_1_after_log.png"))

    # Overlay figure: circles on max projection
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(max_proj_norm, cmap="gray", vmin=0, vmax=1)
    cmap_ov = matplotlib.colormaps.get_cmap("tab20")
    for i, (y, x, r) in enumerate(blobs):
        circle = plt.Circle((x, y), r, color=cmap_ov(i % 20), fill=False, linewidth=0.8)
        ax.add_patch(circle)
        ax.text(x, y - r - 2, str(i + 1), color="yellow", fontsize=5,
                ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Max projection — {N_cells} cells (iterative LoG)", fontsize=14)
    ax.axis("off")
    out_blobs_overlay = out("blobs_overlay_all.png")
    fig.savefig(out_blobs_overlay, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_blobs_overlay}")

    # ── STEP 3: Generate cell masks ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating cell masks")
    print("=" * 60)

    cell_masks = np.zeros((N_cells, H, W), dtype=bool)
    for i in range(1, N_cells + 1):
        cell_masks[i - 1] = labeled_mask == i

    out_masks   = out("cell_masks.npy")
    out_labeled = out("labeled_mask.npy")
    np.save(out_masks,   cell_masks)
    np.save(out_labeled, labeled_mask)
    print(f"cell_masks shape : {cell_masks.shape}  → {out_masks}")
    print(f"labeled_mask shape: {labeled_mask.shape} → {out_labeled}")

    # ── STEP 4: Extract fluorescence traces ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Extracting fluorescence traces")
    print("=" * 60)

    fluorescence_traces = np.zeros((N_frames, N_cells), dtype=np.float64)
    mask_pixels = [np.where(cell_masks[i]) for i in range(N_cells)]

    for f in range(N_frames):
        frame_f = frames[f].astype(np.float64)
        for i, (rows, cols) in enumerate(mask_pixels):
            if len(rows) > 0:
                fluorescence_traces[f, i] = frame_f[rows, cols].mean()
        if (f + 1) % 200 == 0 or f == 0:
            print(f"  Processed frame {f + 1}/{N_frames}")

    out_traces = out("fluorescence_traces.npy")
    np.save(out_traces, fluorescence_traces)
    print(f"fluorescence_traces shape: {fluorescence_traces.shape} → {out_traces}")

    # Waterfall traces plot
    trace_min   = fluorescence_traces.min(axis=0)
    trace_max   = fluorescence_traces.max(axis=0)
    trace_range = np.where(trace_max > trace_min, trace_max - trace_min, 1.0)
    traces_norm = (fluorescence_traces - trace_min) / trace_range

    row_height = 1.2
    fig, ax    = plt.subplots(figsize=(14, max(8, N_cells * 0.35)))
    cmap       = matplotlib.colormaps.get_cmap("tab20")
    frames_x   = np.arange(N_frames)
    for i in range(N_cells):
        offset = (N_cells - 1 - i) * row_height
        ax.plot(frames_x, traces_norm[:, i] + offset,
                color=cmap(i % 20), linewidth=0.6, alpha=0.85)
        ax.text(-20, offset + 0.4, str(i + 1), fontsize=5,
                va="center", ha="right", color=cmap(i % 20))
    ax.set_xlim(-30, N_frames)
    ax.set_ylim(-0.2, N_cells * row_height)
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    ax.set_title(
        f"Fluorescence traces — {N_cells} cells "
        "(vertically stacked, each normalised to its own range)"
    )
    out_traces_png = out("fluorescence_traces.png")
    fig.savefig(out_traces_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_traces_png}")

    # ── STEP 5: Generate overlay video ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Generating overlay video")
    print("=" * 60)

    out_video = out("blobs_overlay_all.mp4")

    # Pre-compute draw parameters once
    draw_cells = [
        (int(round(x)), int(round(y)), max(1, int(round(r))), i + 1)
        for i, (y, x, r) in enumerate(blobs)
    ]

    flat   = frames.ravel()
    p_low  = float(np.percentile(flat, 1))
    p_high = float(np.percentile(flat, 99))

    writer = imageio.get_writer(out_video, fps=10, codec="h264", quality=7,
                                format="FFMPEG", macro_block_size=1)
    for f in range(N_frames):
        frame_f    = frames[f].astype(np.float64)
        norm       = np.clip((frame_f - p_low) / (p_high - p_low + 1e-12), 0, 1)
        uint8_frame = (norm * 255).astype(np.uint8)
        bgr        = cv2.cvtColor(uint8_frame, cv2.COLOR_GRAY2BGR)

        for (cx, cy, cr, label) in draw_cells:
            cv2.circle(bgr, (cx, cy), cr, (0, 255, 0), 1)
            cv2.putText(bgr, str(label), (cx - 5, cy - cr - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if (f + 1) % 100 == 0 or f == 0:
            print(f"  Encoded frame {f + 1}/{N_frames}")

    writer.close()
    print(f"Saved: {out_video}")

    # ── STEP 6: Validation summary figure (4 panels) ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Saving validation summary figure")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    frame0   = frames[0].astype(np.float64)
    f0_norm  = (frame0 - frame0.min()) / (frame0.max() - frame0.min() + 1e-12)
    axes[0, 0].imshow(f0_norm, cmap="gray")
    axes[0, 0].set_title("(a) Raw frame 0", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(max_proj_norm, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("(b) Max projection (normalized)", fontsize=13)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(max_proj_norm, cmap="gray", vmin=0, vmax=1)
    for i, (y, x, r) in enumerate(blobs):
        circle = plt.Circle((x, y), r, color="lime", fill=False, linewidth=0.8)
        axes[1, 0].add_patch(circle)
        axes[1, 0].text(x, y - r - 2, str(i + 1), color="yellow", fontsize=4,
                        ha="center", va="bottom", fontweight="bold")
    axes[1, 0].set_title(f"(c) All {N_cells} detected cells (iterative LoG)", fontsize=13)
    axes[1, 0].axis("off")

    cmap20 = plt.cm.get_cmap("tab20", min(N_cells, 20))
    for i in range(N_cells):
        axes[1, 1].plot(fluorescence_traces[:, i], color=cmap20(i % 20),
                        linewidth=0.4, alpha=0.6)
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Mean intensity")
    axes[1, 1].set_title(f"(d) Fluorescence traces ({N_cells} cells)", fontsize=13)

    plt.suptitle(
        f"Cell imaging pipeline — {N_cells} cells\n"
        f"Iterative LoG  |  gauss_sigma={PARAMS['gauss_sigma']}  "
        f"|  min_sigma={PARAMS['min_sigma']:.1f}  "
        f"|  max_sigma={PARAMS['max_sigma']:.1f}  "
        f"|  threshold={PARAMS['blob_threshold']}  "
        f"|  overlap={PARAMS['overlap']}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out_summary = out("validation_summary.png")
    fig.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_summary}")

    # ── STEP 7: Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Final summary")
    print("=" * 60)

    print(f"\nCells detected: {N_cells}")
    print("\nParameters used:")
    for k, v in PARAMS.items():
        print(f"  {k:<25s}: {v}")

    output_files = [
        out_blobs_overlay, out_masks, out_labeled,
        out_traces, out_traces_png, out_video, out_summary,
    ]
    print("\nSaved files:")
    for f_path in output_files:
        if os.path.exists(f_path):
            size_mb = os.path.getsize(f_path) / 1_048_576
            print(f"  {os.path.basename(f_path):40s}  {size_mb:.2f} MB")
        else:
            print(f"  {os.path.basename(f_path):40s}  [NOT FOUND]")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
