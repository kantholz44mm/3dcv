from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "images"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 – Load data
# ---------------------------------------------------------------------------

def load_calibration(path: Path) -> dict:
    calib = {}
    with np.load(path) as f:
        for key in f.files:
            calib[key] = f[key]
    return calib


def read_pfm(path: Path):
    """Read PFM file, return (image_float32, scale)."""
    with path.open("rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        dims = f.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)
        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, endian + "f")
        shape = (h, w, channels) if channels == 3 else (h, w)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data.astype(np.float32), scale


def load_images():
    im0 = cv2.imread(str(IMAGES / "artroom_im0.png"))
    im1 = cv2.imread(str(IMAGES / "artroom_im1.png"))
    if im0 is None or im1 is None:
        raise FileNotFoundError("Could not load stereo images from images/")
    return im0, im1


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Step 2 – Compute disparity
# ---------------------------------------------------------------------------

def compute_disparity_bm(gray_l, gray_r, ndisp: int) -> np.ndarray:
    num_disp = int(np.ceil(ndisp / 16.0) * 16)
    matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp[disp < 0] = np.nan
    return disp


def _make_sgbm(ndisp: int) -> cv2.StereoSGBM:
    num_disp = int(np.ceil(ndisp / 16.0) * 16)
    win = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=win,
        P1=8 * win * win,
        P2=32 * win * win,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity_sgbm(gray_l, gray_r, ndisp: int) -> np.ndarray:
    matcher = _make_sgbm(ndisp)
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp[disp < 0] = np.nan
    return disp


def compute_disparity_wls(gray_l, gray_r, ndisp: int) -> tuple[np.ndarray, np.ndarray]:
    """SGBM + WLS filter. Returns (filtered_disparity, confidence_map)."""
    left_matcher  = _make_sgbm(ndisp)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_raw  = left_matcher.compute(gray_l, gray_r)
    right_raw = right_matcher.compute(gray_r, gray_l)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000)
    wls.setSigmaColor(1.5)

    filtered = wls.filter(left_raw, gray_l, disparity_map_right=right_raw)
    confidence = wls.getConfidenceMap()

    disp = filtered.astype(np.float32) / 16.0
    disp[disp < 0] = np.nan
    return disp, confidence


def colorize_disparity(disp: np.ndarray) -> np.ndarray:
    """Return BGR image of disparity for display / saving."""
    vis = np.nan_to_num(disp, nan=0.0)
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
    vis = np.uint8(vis)
    return cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)


# ---------------------------------------------------------------------------
# Step 3 – Evaluate disparity
# ---------------------------------------------------------------------------

def _valid_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)


def mean_absolute_error(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = _valid_mask(gt, pred)
    return float(np.mean(np.abs(gt[mask] - pred[mask])))


def bad_pixel_ratio(gt: np.ndarray, pred: np.ndarray, threshold: float = 1.0) -> float:
    """Fraction of valid pixels where |gt - pred| > threshold."""
    mask = _valid_mask(gt, pred)
    bad = np.abs(gt[mask] - pred[mask]) > threshold
    return float(bad.mean())


# ---------------------------------------------------------------------------
# Step 4 – Compute depth
# ---------------------------------------------------------------------------

def disparity_to_depth(disp: np.ndarray, focal_px: float, baseline_mm: float) -> np.ndarray:
    depth = np.full_like(disp, np.nan)
    valid = (disp > 0) & np.isfinite(disp)
    depth[valid] = focal_px * baseline_mm / disp[valid]
    return depth


# ---------------------------------------------------------------------------
# Step 5 – Export point cloud
# ---------------------------------------------------------------------------

def build_q_matrix(calib: dict) -> np.ndarray:
    cam0  = calib["cam0"]
    f     = float(cam0[0, 0])
    cx    = float(cam0[0, 2])
    cy    = float(cam0[1, 2])
    B     = float(calib["baseline"])
    doffs = float(calib["doffs"])
    return np.array([
        [1,  0,  0,        -cx],
        [0,  1,  0,        -cy],
        [0,  0,  0,          f],
        [0,  0, -1.0 / B, doffs / B],
    ], dtype=np.float64)


def remove_statistical_outliers(
    pts: np.ndarray, cols: np.ndarray, k: int = 20, n_sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Remove points whose mean k-NN distance exceeds mean + n_sigma * std."""
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1, workers=-1)  # k+1: first hit is self
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + n_sigma * mean_dists.std()
    keep = mean_dists < threshold
    removed = (~keep).sum()
    print(f"    outlier removal: kept {keep.sum():,} / {len(pts):,} points "
          f"(removed {removed:,}, {100*removed/len(pts):.1f}%)")
    return pts[keep], cols[keep]


def export_ply(disp: np.ndarray, color_bgr: np.ndarray, Q: np.ndarray,
               vmin: float, vmax: float, out_path: Path,
               clean: bool = False, k: int = 20, n_sigma: float = 2.0):
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    mask = (disp >= vmin) & (disp <= vmax) & np.isfinite(disp)

    pts  = points_3d[mask]
    cols = color_bgr[mask][:, ::-1]  # BGR → RGB

    finite = np.all(np.isfinite(pts), axis=1)
    pts  = pts[finite]
    cols = cols[finite]

    if clean:
        pts, cols = remove_statistical_outliers(pts, cols, k=k, n_sigma=n_sigma)

    with out_path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"  Saved {len(pts):,} points → {out_path}")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def save_figure(fig, name: str):
    path = DATA / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure → {path}")
    plt.close(fig)


def plot_stereo_pair(im0_bgr, im1_bgr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, img, title in zip(axes, [im0_bgr, im1_bgr], ["Left (im0)", "Right (im1)"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "stereo_pair.png")


def plot_disparity_comparison(gt_disp, maps: list, vmin: float, vmax: float):
    """maps: list of (label, disparity_array)"""
    ncols = 1 + len(maps)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    all_maps = [("Ground truth", gt_disp)] + maps
    for ax, (title, d) in zip(axes, all_maps):
        im = ax.imshow(d, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, label="disparity (px)")
    fig.tight_layout()
    save_figure(fig, "disparity_comparison.png")


def plot_error_maps(gt_disp, maps: list):
    """maps: list of (label, disparity_array)"""
    fig, axes = plt.subplots(1, len(maps), figsize=(7 * len(maps), 5))
    if len(maps) == 1:
        axes = [axes]
    for ax, (label, pred) in zip(axes, maps):
        mask = _valid_mask(gt_disp, pred)
        err = np.full_like(gt_disp, np.nan)
        err[mask] = np.abs(gt_disp[mask] - pred[mask])
        im = ax.imshow(err, cmap="hot", vmin=0, vmax=10)
        ax.set_title(f"Absolute error – {label}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, label="|error| (px)")
    fig.tight_layout()
    save_figure(fig, "disparity_error_maps.png")


def plot_depth_map(depth_mm: np.ndarray, label: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    d_vis = np.clip(depth_mm, 0, np.nanpercentile(depth_mm, 98))
    im = ax.imshow(d_vis, cmap="viridis")
    ax.set_title(f"Depth map ({label}) [mm]")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, label="depth (mm)")
    fig.tight_layout()
    save_figure(fig, f"depth_map_{label.lower().replace(' ', '_')}.png")


def plot_wls_confidence(confidence: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(confidence, cmap="gray")
    ax.set_title("WLS confidence map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    save_figure(fig, "wls_confidence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Loading data ===")
    im0, im1 = load_images()
    calib = load_calibration(DATA / "artroom_calib.npz")
    gt_disp, _ = read_pfm(IMAGES / "disp0.pfm")
    gt_disp[gt_disp == 0] = np.nan  # 0 marks invalid pixels in the GT

    focal_px = float(calib["cam0"][0, 0])
    baseline = float(calib["baseline"])
    ndisp    = int(calib["ndisp"])
    vmin     = float(calib["vmin"])
    vmax     = float(calib["vmax"])

    print(f"  focal length : {focal_px:.2f} px")
    print(f"  baseline     : {baseline:.2f} mm")
    print(f"  ndisp        : {ndisp}")
    print(f"  valid range  : [{vmin}, {vmax}] px")

    print("\n=== Visualising stereo pair ===")
    plot_stereo_pair(im0, im1)

    gray0 = to_gray(im0)
    gray1 = to_gray(im1)

    print("\n=== Step 2: Computing disparity maps ===")
    bm_disp   = compute_disparity_bm(gray0, gray1, ndisp)
    print("  StereoBM   done")
    sgbm_disp = compute_disparity_sgbm(gray0, gray1, ndisp)
    print("  StereoSGBM done")
    wls_disp, confidence = compute_disparity_wls(gray0, gray1, ndisp)
    print("  SGBM+WLS   done")

    for label, disp in [("bm", bm_disp), ("sgbm", sgbm_disp), ("wls", wls_disp)]:
        cv2.imwrite(str(DATA / f"disparity_{label}.png"), colorize_disparity(disp))

    print("\n=== Step 3: Evaluating disparity ===")
    all_maps = [("StereoBM", bm_disp), ("StereoSGBM", sgbm_disp), ("SGBM+WLS", wls_disp)]

    # Restrict WLS evaluation to pixels where SGBM also has a valid prediction,
    # so the metrics are not inflated by the extra hole-filling WLS does.
    sgbm_coverage = np.isfinite(sgbm_disp) & (sgbm_disp > 0)
    wls_disp_eval = wls_disp.copy()
    wls_disp_eval[~sgbm_coverage] = np.nan

    eval_maps = [("StereoBM", bm_disp), ("StereoSGBM", sgbm_disp), ("SGBM+WLS", wls_disp_eval)]
    for label, pred in eval_maps:
        mae  = mean_absolute_error(gt_disp, pred)
        bad1 = bad_pixel_ratio(gt_disp, pred, threshold=1.0) * 100
        bad2 = bad_pixel_ratio(gt_disp, pred, threshold=2.0) * 100
        print(f"  {label:12s}  MAE={mae:.2f} px  bad>1px={bad1:.1f}%  bad>2px={bad2:.1f}%")

    plot_disparity_comparison(gt_disp, all_maps, vmin, vmax)
    plot_error_maps(gt_disp, all_maps)
    plot_wls_confidence(confidence)

    print("\n=== Step 4: Computing depth maps ===")
    for label, disp in all_maps:
        depth = disparity_to_depth(disp, focal_px, baseline)
        plot_depth_map(depth, label)

    print("\n=== Step 5: Exporting point clouds ===")
    Q = build_q_matrix(calib)
    for label, disp in all_maps:
        slug = label.lower().replace("+", "_").replace(" ", "_")
        print(f"  {label} (raw):")
        export_ply(disp, im0, Q, vmin, vmax, DATA / f"pointcloud_{slug}.ply", clean=False)
        print(f"  {label} (cleaned):")
        export_ply(disp, im0, Q, vmin, vmax, DATA / f"pointcloud_{slug}_clean.ply", clean=True)

    print("\nDone. All outputs saved to data/")


if __name__ == "__main__":
    main()
