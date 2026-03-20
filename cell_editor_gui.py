#!/usr/bin/env python3
"""
cell_editor_gui.py — Interactive cell label editor for fluorescence imaging.

Usage
-----
    python cell_editor_gui.py path/to/imaging_file.tif
    python cell_editor_gui.py path/to/imaging_file.tif path/to/labeled_mask.npy

If the labeled_mask path is omitted the GUI auto-discovers the most recent
output_YYYY-MM-DD/labeled_mask.npy in the same directory as the TIFF.

Controls
--------
Delete mode  [d]   : left-click anywhere inside a cell to remove it
Add mode     [a]   : click 4 boundary points to define a new cell
                     (e.g. top / bottom / left-edge / right-edge)
                     A dashed ellipse preview updates after the 2nd click.
                     The 4th click finalises the cell.
                     Press Escape to cancel an in-progress add.
Undo         [z]   : undo last add or delete (up to 50 steps)
Save         [s]   : run full pipeline → output_YYYY-MM-DD/ folder
"""

import argparse
import glob
import os
import sys
import threading
from datetime import date

# ── portable import of detection module ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import imageio
import matplotlib
matplotlib.use("MacOSX")      # native macOS interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.widgets import Button
import numpy as np
import tifffile
from skimage.draw import ellipse as sk_ellipse
from skimage.measure import regionprops

from detection import fit_ellipse_pca


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive cell label editor for fluorescence imaging."
    )
    parser.add_argument("tiff", help="Path to the input TIFF imaging file.")
    parser.add_argument(
        "mask",
        nargs="?",
        default=None,
        help=(
            "Path to labeled_mask.npy to load.  "
            "If omitted, the most recent output_YYYY-MM-DD/labeled_mask.npy "
            "in the TIFF directory is used."
        ),
    )
    return parser.parse_args()


# ── module-level helpers ───────────────────────────────────────────────────────
def _find_latest_labeled_mask(base_dir):
    """
    Return the path to the most recent labeled_mask.npy.

    Scans output_YYYY-MM-DD/ sub-folders of base_dir (lexicographic descending
    = newest date first) and returns the first match.  Falls back to
    base_dir/labeled_mask.npy if no output folder exists yet.
    """
    candidates = sorted(
        glob.glob(os.path.join(base_dir, "output_*", "labeled_mask.npy")),
        reverse=True,
    )
    if candidates:
        print(f"[GUI] Loading labeled mask from: {candidates[0]}")
        return candidates[0]
    fallback = os.path.join(base_dir, "labeled_mask.npy")
    print(f"[GUI] No output_* folder found — loading from: {fallback}")
    return fallback


def _rasterise_ellipse(cy, cx, semi_major, semi_minor, angle_deg, H, W):
    """
    Return (rr, cc) pixel coordinates of all pixels inside a fitted ellipse.

    Parameters
    ----------
    cy, cx       : float   Centre in image (row, col) coordinates.
    semi_major   : float   Semi-major axis length in pixels.
    semi_minor   : float   Semi-minor axis length in pixels.
    angle_deg    : float   Rotation of major axis from x-axis, CCW, degrees.
    H, W         : int     Image dimensions (rows, cols).

    Returns
    -------
    rr, cc : np.ndarray   Integer row and column indices inside the ellipse.
    """
    rr, cc = sk_ellipse(
        int(round(cy)), int(round(cx)),
        max(1, int(round(semi_major))),
        max(1, int(round(semi_minor))),
        rotation=-np.radians(angle_deg),   # skimage rotation convention is CW
        shape=(H, W),
    )
    return rr, cc


# ── main editor class ──────────────────────────────────────────────────────────
class CellEditor:
    def __init__(self, tiff_path, mask_path=None):
        self.tiff_path = tiff_path
        self.base_dir  = os.path.dirname(tiff_path)

        # ── load TIFF ─────────────────────────────────────────────────────────
        print("Loading TIFF …")
        self.frames = tifffile.imread(tiff_path)
        if self.frames.ndim == 4:
            self.frames = self.frames[:, 0, :, :]
        self.N_frames, self.H, self.W = self.frames.shape
        print(f"  {self.N_frames} frames  {self.H}×{self.W}")

        # ── display image: percentile-normalised max projection ───────────────
        raw_max = self.frames.max(axis=0).astype(np.float64)
        p1, p99 = np.percentile(raw_max, [1, 99])
        self.display_img = np.clip((raw_max - p1) / (p99 - p1), 0, 1)

        # ── load labeled mask ─────────────────────────────────────────────────
        if mask_path is None:
            mask_path = _find_latest_labeled_mask(self.base_dir)
        else:
            print(f"[GUI] Loading labeled mask from: {mask_path}")
        self.labeled = np.load(mask_path).astype(np.int32)

        # ── shape_params: [cy, cx, semi_major, semi_minor, angle_deg] ─────────
        self.shape_params = self._build_shape_params(self.labeled)
        print(f"  {len(self.shape_params)} cells loaded")

        # ── GUI interaction state ─────────────────────────────────────────────
        self.mode       = "delete"
        self.undo_stack = []

        # 4-click ellipse add state
        self.ell_points  = []
        self.ell_dots    = []
        self.ell_preview = None

        self._build_gui()

    # ── shape param helpers ────────────────────────────────────────────────────
    def _build_shape_params(self, labeled):
        """
        Derive a shape_params dict from a labeled mask using regionprops.

        Every cell is stored as [cy, cx, semi_major, semi_minor, angle_deg]
        where angle_deg is CCW from the x-axis (matplotlib Ellipse convention).
        LoG-detected cells are roughly circular so semi_major ≈ semi_minor;
        PCA-added cells retain their true ellipse geometry.

        Parameters
        ----------
        labeled : np.int32 (H, W)

        Returns
        -------
        dict  { cell_id (int) → np.ndarray [cy, cx, semi_major, semi_minor, angle_deg] }
        """
        sp = {}
        for region in regionprops(labeled):
            cy, cx     = region.centroid
            semi_major = region.major_axis_length / 2.0
            semi_minor = region.minor_axis_length / 2.0
            # regionprops orientation: angle from row-axis (vertical) CCW, radians
            # matplotlib Ellipse angle: CCW from x-axis (horizontal), degrees
            angle_deg  = 90.0 - np.degrees(region.orientation)
            sp[region.label] = np.array([cy, cx, semi_major, semi_minor, angle_deg])
        return sp

    def _next_cell_id(self):
        existing = list(self.shape_params.keys())
        return max(existing) + 1 if existing else 1

    # ── GUI construction ───────────────────────────────────────────────────────
    def _build_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.10)

        self.ax.imshow(self.display_img, cmap="gray", origin="upper")
        self.patches = []
        self._redraw()

        ax_del  = plt.axes([0.04, 0.02, 0.22, 0.06])
        ax_add  = plt.axes([0.28, 0.02, 0.22, 0.06])
        ax_undo = plt.axes([0.52, 0.02, 0.18, 0.06])
        ax_save = plt.axes([0.72, 0.02, 0.25, 0.06])

        self.btn_del  = Button(ax_del,  "Delete  [d]",         color="lightcoral")
        self.btn_add  = Button(ax_add,  "Add  [a]",            color="lightgreen")
        self.btn_undo = Button(ax_undo, "Undo  [z]")
        self.btn_save = Button(ax_save, "Save & Process  [s]", color="gold")

        self.btn_del.on_clicked( lambda _: self._set_mode("delete"))
        self.btn_add.on_clicked( lambda _: self._set_mode("add"))
        self.btn_undo.on_clicked(lambda _: self._undo())
        self.btn_save.on_clicked(lambda _: self._save_and_process())

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)

        self._update_button_highlights()
        plt.show()

    # ── drawing ────────────────────────────────────────────────────────────────
    def _redraw(self):
        """Redraw all cell outlines (ellipses) and update the title."""
        for p in self.patches:
            p.remove()
        self.patches.clear()

        cmap     = matplotlib.colormaps.get_cmap("tab20")
        cell_ids = sorted(self.shape_params.keys())

        for idx, cid in enumerate(cell_ids):
            cy, cx, sm, sn, ang = self.shape_params[cid]
            color = cmap(idx % 20)
            patch = mpatches.Ellipse(
                (cx, cy), width=2 * sm, height=2 * sn, angle=ang,
                fill=False, edgecolor=color, linewidth=1.0)
            self.ax.add_patch(patch)
            r_label = max(sm, sn)
            txt = self.ax.text(cx, cy - r_label - 2, str(idx + 1),
                               color="white", fontsize=5,
                               ha="center", va="bottom")
            self.patches.extend([patch, txt])

        n = len(cell_ids)
        if self.mode == "add" and self.ell_points:
            k    = len(self.ell_points)
            hint = f"Add: {k}/4 points placed — click {4 - k} more (Esc to cancel)"
        elif self.mode == "add":
            hint = "Add: click 4 boundary points to define a new cell"
        else:
            hint = "Delete: click inside a cell to remove it"

        self.ax.set_title(f"Cell Editor  |  {n} cells  |  {hint}")
        self.fig.canvas.draw_idle()

    # ── mode ──────────────────────────────────────────────────────────────────
    def _set_mode(self, mode):
        self._cancel_add()
        self.mode = mode
        self._update_button_highlights()
        self._redraw()

    def _update_button_highlights(self):
        self.btn_del.ax.set_facecolor(
            "tomato"    if self.mode == "delete" else "lightcoral")
        self.btn_add.ax.set_facecolor(
            "limegreen" if self.mode == "add"    else "lightgreen")
        self.fig.canvas.draw_idle()

    # ── undo ──────────────────────────────────────────────────────────────────
    def _push_undo(self):
        snap = (self.labeled.copy(),
                {k: v.copy() for k, v in self.shape_params.items()})
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self):
        if not self.undo_stack:
            print("[GUI] Nothing to undo.")
            return
        self.labeled, self.shape_params = self.undo_stack.pop()
        self._redraw()

    # ── cancel in-progress add ────────────────────────────────────────────────
    def _cancel_add(self):
        for d in self.ell_dots:
            d.remove()
        self.ell_dots.clear()
        if self.ell_preview is not None:
            self.ell_preview.remove()
            self.ell_preview = None
        self.ell_points.clear()
        self.fig.canvas.draw_idle()

    # ── keyboard / mouse events ───────────────────────────────────────────────
    def _on_key(self, event):
        if   event.key == "d":       self._set_mode("delete")
        elif event.key == "a":       self._set_mode("add")
        elif event.key == "z":       self._undo()
        elif event.key == "s":       self._save_and_process()
        elif event.key == "escape":  self._cancel_add(); self._redraw()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        y, x = event.ydata, event.xdata

        if self.mode == "delete":
            self._try_delete(y, x)

        elif self.mode == "add":
            self.ell_points.append((y, x))
            dot, = self.ax.plot(x, y, "c+", markersize=8, markeredgewidth=1.5)
            self.ell_dots.append(dot)

            n = len(self.ell_points)
            if n >= 2:
                self._update_ellipse_preview()
            self._redraw()
            if n == 4:
                self._finalise_ellipse()

    # ── 4-click ellipse helpers ────────────────────────────────────────────────
    def _update_ellipse_preview(self):
        """
        Display a live dashed Ellipse preview fitted to the clicks collected
        so far (minimum 2 points required).
        """
        cy, cx, sm, sn, ang = fit_ellipse_pca(self.ell_points)
        if self.ell_preview is not None:
            self.ell_preview.remove()
        self.ell_preview = mpatches.Ellipse(
            (cx, cy), width=2 * sm, height=2 * sn, angle=ang,
            fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--")
        self.ax.add_patch(self.ell_preview)
        self.fig.canvas.draw_idle()

    def _finalise_ellipse(self):
        """
        Called after the 4th boundary click.

        Fits the final PCA ellipse, rasterises it into the labeled mask, and
        commits the true ellipse geometry (cy, cx, sm, sn, ang) to shape_params
        so the display outline exactly matches the pixel mask.  Cleans up all
        preview artefacts.
        """
        cy, cx, sm, sn, ang = fit_ellipse_pca(self.ell_points)

        for d in self.ell_dots:
            d.remove()
        self.ell_dots.clear()
        if self.ell_preview is not None:
            self.ell_preview.remove()
            self.ell_preview = None
        self.ell_points.clear()

        if sm < 1 or sn < 1:
            self.fig.canvas.draw_idle()
            return

        self._push_undo()
        new_id = self._next_cell_id()
        rr, cc = _rasterise_ellipse(cy, cx, sm, sn, ang, self.H, self.W)
        self.labeled[rr, cc] = new_id

        # Store actual ellipse geometry so the display outline matches the mask
        self.shape_params[new_id] = np.array([cy, cx, sm, sn, ang])
        self._redraw()

    # ── delete ────────────────────────────────────────────────────────────────
    def _try_delete(self, y, x):
        """
        Delete the cell whose mask pixel is at image coordinate (y, x).

        Hit-tests directly against the labeled mask so it works correctly for
        both circular and arbitrary-shaped ROIs.
        """
        yi  = int(np.clip(round(y), 0, self.H - 1))
        xi  = int(np.clip(round(x), 0, self.W - 1))
        cid = int(self.labeled[yi, xi])
        if cid == 0:
            print("[GUI] Clicked background — no cell there.")
            return
        self._push_undo()
        self.labeled[self.labeled == cid] = 0
        del self.shape_params[cid]
        print(f"[GUI] Deleted cell id={cid}")
        self._redraw()

    # ── save & process ────────────────────────────────────────────────────────
    def _save_and_process(self):
        print("\n[GUI] Starting Save & Process …")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        labeled      = self.labeled.copy()
        shape_params = {k: v.copy() for k, v in self.shape_params.items()}
        frames       = self.frames
        N_frames     = self.N_frames
        H, W         = self.H, self.W

        cell_ids = sorted(shape_params.keys())
        N_cells  = len(cell_ids)

        today   = date.today().strftime("%Y-%m-%d")
        out_dir = os.path.join(self.base_dir, f"output_{today}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Output → {out_dir}  ({N_cells} cells)")

        def out(fname):
            return os.path.join(out_dir, fname)

        # ── 1. save label metadata ─────────────────────────────────────────────
        np.save(out("labeled_mask.npy"),  labeled)
        shapes_arr = np.array([shape_params[cid] for cid in cell_ids])
        np.save(out("shape_params.npy"),  shapes_arr)
        np.save(out("cell_ids.npy"),      np.array(cell_ids))

        # ── 2. cell masks ──────────────────────────────────────────────────────
        print("  Building cell masks …")
        cell_masks = np.zeros((N_cells, H, W), dtype=bool)
        for i, cid in enumerate(cell_ids):
            cell_masks[i] = labeled == cid
        np.save(out("cell_masks.npy"), cell_masks)

        # ── 3. fluorescence traces ─────────────────────────────────────────────
        print("  Extracting fluorescence traces …")
        traces      = np.zeros((N_frames, N_cells), dtype=np.float64)
        mask_pixels = [np.where(cell_masks[i]) for i in range(N_cells)]
        for f in range(N_frames):
            frame_f = frames[f].astype(np.float64)
            for i, (rr, cc) in enumerate(mask_pixels):
                traces[f, i] = frame_f[rr, cc].mean() if len(rr) > 0 else 0.0
            if f % 200 == 0:
                print(f"    frame {f}/{N_frames}")
        np.save(out("fluorescence_traces.npy"), traces)

        # ── 4. traces figure (stacked waterfall) ──────────────────────────────
        print("  Plotting traces …")
        tr_min  = traces.min(axis=0)
        tr_max  = traces.max(axis=0)
        tr_rng  = np.where(tr_max > tr_min, tr_max - tr_min, 1.0)
        tr_norm = (traces - tr_min) / tr_rng

        row_h = 1.2
        fig2  = Figure(figsize=(14, max(8, N_cells * 0.35)))
        ax2   = fig2.add_subplot(111)
        cmap  = matplotlib.colormaps.get_cmap("tab20")
        xs    = np.arange(N_frames)
        for i in range(N_cells):
            offset = (N_cells - 1 - i) * row_h
            ax2.plot(xs, tr_norm[:, i] + offset,
                     color=cmap(i % 20), linewidth=0.6, alpha=0.85)
            ax2.text(-20, offset + 0.4, str(i + 1),
                     fontsize=5, va="center", ha="right", color=cmap(i % 20))
        ax2.set_xlim(-30, N_frames)
        ax2.set_ylim(-0.2, N_cells * row_h)
        ax2.set_xlabel("Frame")
        ax2.set_yticks([])
        ax2.set_title(
            f"Fluorescence traces — {N_cells} cells "
            f"(stacked, each normalised to its own range)")
        fig2.savefig(out("fluorescence_traces.png"), dpi=150, bbox_inches="tight")

        # ── 5. overlay video ───────────────────────────────────────────────────
        print("  Generating overlay video …")
        out_video = out("blobs_overlay.mp4")
        flat  = frames.ravel()
        p_low = float(np.percentile(flat, 1))
        p_hi  = float(np.percentile(flat, 99))

        writer = imageio.get_writer(out_video, fps=10, codec="h264",
                                    quality=7, format="FFMPEG",
                                    macro_block_size=1)
        for f in range(N_frames):
            frame_f = frames[f].astype(np.float64)
            u8  = np.clip((frame_f - p_low) / (p_hi - p_low) * 255,
                          0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            for i, cid in enumerate(cell_ids):
                cy, cx, sm, sn, ang = shape_params[cid]
                cx_i      = int(round(cx))
                cy_i      = int(round(cy))
                ax_sm     = max(1, int(round(sm)))
                ax_sn     = max(1, int(round(sn)))
                cv2_angle = int(round(-ang % 360))
                cv2.ellipse(rgb, (cx_i, cy_i), (ax_sm, ax_sn),
                            cv2_angle, 0, 360, (0, 255, 0), 1)
                label_y = cy_i - ax_sm - 2
                cv2.putText(rgb, str(i + 1), (cx_i - 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)
            writer.append_data(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            if f % 200 == 0:
                print(f"    video frame {f}/{N_frames}")
        writer.close()

        # ── summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"DONE — {N_cells} cells — outputs in {out_dir}/")
        for fname in ["labeled_mask.npy", "shape_params.npy", "cell_ids.npy",
                      "cell_masks.npy", "fluorescence_traces.npy",
                      "fluorescence_traces.png", "blobs_overlay.mp4"]:
            fpath = out(fname)
            if os.path.exists(fpath):
                mb = os.path.getsize(fpath) / 1_048_576
                print(f"  {fname:40s}  {mb:.2f} MB")
        print(f"{'='*60}\n")


# ── entry point ────────────────────────────────────────────────────────────────
def launch_editor():
    args = _parse_args()
    tiff_path = os.path.abspath(args.tiff)
    if not os.path.isfile(tiff_path):
        sys.exit(f"ERROR: TIFF file not found: {tiff_path}")
    mask_path = os.path.abspath(args.mask) if args.mask else None
    CellEditor(tiff_path, mask_path)


if __name__ == "__main__":
    launch_editor()
