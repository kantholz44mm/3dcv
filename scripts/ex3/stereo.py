"""
3DCV Workshop 03 - Simple Stereo
Stereo disparity estimation, evaluation, depth computation, and point cloud export.

Usage:
    python stereo.py

Dependencies:
    pip install opencv-python numpy

Expected file structure:
    images/artroom_im0.png
    images/artroom_im1.png
    images/disp0.pfm
    data/artroom_calib.npz
"""

from pathlib import Path
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Step 1 - Load stereo images and calibration data
# ---------------------------------------------------------------------------

def load_images(left_path: Path, right_path: Path):
    left  = cv2.imread(str(left_path),  cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    left_color = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Could not load images: {left_path}, {right_path}")
    return left, right, left_color


def load_calibration(path: Path) -> dict:
    calib = {}
    with np.load(path) as data:
        for key in data.files:
            calib[key] = data[key]
    f  = float(calib["cam0"][0, 0])
    B  = float(calib["baseline"])
    print("Calibration keys:", list(calib.keys()))
    print(f"  focal length (f): {f}")
    print(f"  baseline (B):     {B}")
    return calib


def read_pfm(path: Path):
    """Read PFM file, return (image, scale)."""
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


# ---------------------------------------------------------------------------
# Step 2 - Compute disparity map
# ---------------------------------------------------------------------------

def compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray, ndisp: int = 170) -> np.ndarray:
    num_disp = int(np.ceil(ndisp / 16.0) * 16)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=5,
        P1=8  * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def visualize_disparity(disparity: np.ndarray, title: str = "Disparity"):
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_PLASMA)
    cv2.imshow(title, disp_vis)
    return disp_vis


# ---------------------------------------------------------------------------
# Step 3 - Evaluate disparity map
# ---------------------------------------------------------------------------

def mean_absolute_error(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(gt) & (gt > 0)
    return float(np.mean(np.abs(gt[mask] - pred[mask])))


def bad_pixel_ratio(gt: np.ndarray, pred: np.ndarray, threshold: float = 2.0) -> float:
    """Percentage of pixels where |gt - pred| > threshold."""
    mask = ~np.isnan(gt) & (gt > 0)
    bad  = np.abs(gt[mask] - pred[mask]) > threshold
    return float(np.mean(bad)) * 100.0


def evaluate(gt: np.ndarray, pred: np.ndarray):
    mae = mean_absolute_error(gt, pred)
    bad = bad_pixel_ratio(gt, pred, threshold=2.0)
    print(f"  MAE:              {mae:.4f} px")
    print(f"  Bad pixels (>2):  {bad:.2f}%")
    return mae, bad


# ---------------------------------------------------------------------------
# Step 4 - Compute depth map
# ---------------------------------------------------------------------------

def disparity_to_depth(disparity: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    depth = disparity.copy()
    depth[depth <= 0] = np.nan
    depth = (focal * baseline) / depth
    return depth


def visualize_depth(depth: np.ndarray, title: str = "Depth"):
    valid = depth[~np.isnan(depth)]
    lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
    depth_clipped = np.clip(depth, lo, hi)
    depth_vis = cv2.normalize(depth_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imshow(title, depth_vis)
    return depth_vis


# ---------------------------------------------------------------------------
# Step 5 - Export point cloud as PLY
# ---------------------------------------------------------------------------

def export_point_cloud(disparity: np.ndarray, left_color: np.ndarray,
                       calib: dict, out_path: Path):
    f  = float(calib["cam0"][0, 0])
    B  = float(calib["baseline"])
    cx = float(calib["cam0"][0, 2])
    cy = float(calib["cam0"][1, 2])
    doffs = float(calib.get("doffs", 0.0))

    # Build Q matrix for reprojectImageTo3D
    Q = np.float32([
        [1,  0,  0,          -cx],
        [0,  1,  0,          -cy],
        [0,  0,  0,            f],
        [0,  0, -1/B,  doffs/B],
    ])

    vmin = float(calib.get("vmin", 1.0))
    vmax = float(calib.get("vmax", float(np.nanmax(disparity))))

    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Mask: valid disparity range and finite depth
    mask = (
        (disparity >= vmin) &
        (disparity <= vmax) &
        np.isfinite(points_3d[:, :, 2])
    )

    pts    = points_3d[mask]
    colors = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB)[mask]

    print(f"  Exporting {len(pts):,} points to {out_path}")

    with out_path.open("w") as ply:
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write(f"element vertex {len(pts)}\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        ply.write("property uchar red\n")
        ply.write("property uchar green\n")
        ply.write("property uchar blue\n")
        ply.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, colors):
            ply.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

    print(f"  Point cloud saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    left_path  = Path("images/artroom_im0.png")
    right_path = Path("images/artroom_im1.png")
    gt_path    = Path("images/disp0.pfm")
    calib_path = Path("data/artroom_calib.npz")
    ply_path   = Path("output_pointcloud.ply")

    # Step 1
    print("=== Step 1: Loading images and calibration ===")
    left_gray, right_gray, left_color = load_images(left_path, right_path)
    calib = load_calibration(calib_path)
    gt_disp, _ = read_pfm(gt_path)
    print(f"  Image size: {left_gray.shape[1]}x{left_gray.shape[0]}")
    print(f"  GT disparity range: [{np.nanmin(gt_disp):.2f}, {np.nanmax(gt_disp):.2f}]")

    # Step 2
    print("\n=== Step 2: Computing disparity ===")
    ndisp = int(calib.get("ndisp", calib.get("NDISP", 170)))
    disparity = compute_disparity(left_gray, right_gray, ndisp=ndisp)
    print(f"  Computed disparity range: [{disparity.min():.2f}, {disparity.max():.2f}]")

    disp_vis = visualize_disparity(disparity, "Computed Disparity")
    gt_vis   = visualize_disparity(gt_disp,   "Ground Truth Disparity")
    cv2.imwrite("images/ex3_disparity_computed.png", disp_vis)
    cv2.imwrite("images/ex3_disparity_gt.png",       gt_vis)

    # Step 3
    print("\n=== Step 3: Evaluating disparity ===")
    # Resize gt to match computed if needed
    if gt_disp.shape != disparity.shape:
        gt_disp = cv2.resize(gt_disp, (disparity.shape[1], disparity.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    evaluate(gt_disp, disparity)

    # Step 4
    print("\n=== Step 4: Computing depth map ===")
    f = float(calib["cam0"][0, 0])
    B = float(calib["baseline"])
    print(f"  f={f}, B={B}")
    depth = disparity_to_depth(disparity, f, B)
    valid_depth = depth[~np.isnan(depth)]
    print(f"  Depth range: [{np.nanmin(valid_depth):.2f}, {np.nanmax(valid_depth):.2f}]")

    depth_vis = visualize_depth(depth, "Depth Map")
    cv2.imwrite("images/ex3_depth_map.png", depth_vis)

    # Step 5
    print("\n=== Step 5: Exporting point cloud ===")
    export_point_cloud(disparity, left_color, calib, ply_path)

    print("\nDone. Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()