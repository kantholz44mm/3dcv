from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT   = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "images"
DATA   = ROOT / "data"
DATA.mkdir(exist_ok=True)

# tunable parameters
NDISP         = 256   # rounded up to nearest multiple of 16 automatically
BASELINE_CM   = None  # baseline in cm (same units as calibration); None = relative depth
MAX_DEPTH_CM  = None  # hard cap on depth colormap in cm; None = auto (95th percentile)


# -- Step 1: Load images and calibration --

def load_images():
    left  = cv2.imread(str(IMAGES / "my_pair_left.png"))
    right = cv2.imread(str(IMAGES / "my_pair_right.png"))
    if left is None or right is None:
        raise FileNotFoundError(
            f"Stereo pair not found. Expected:\n"
            f"  {IMAGES / 'my_pair_left.png'}\n"
            f"  {IMAGES / 'my_pair_right.png'}"
        )
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same resolution.")
    return left, right


def load_calibration():
    with np.load(ROOT / "calibration.npz") as f:
        K    = f["camera_matrix"]
        dist = f["dist_coeffs"]
    return K, dist


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -- Step 2: Feature detection and matching --

def match_features(gray_left, gray_right):
    sift = cv2.SIFT_create()
    kp_left,  desc_left  = sift.detectAndCompute(gray_left,  None)
    kp_right, desc_right = sift.detectAndCompute(gray_right, None)

    matcher     = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_left, desc_right, k=2)
    good        = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    pts_left  = np.array([kp_left[m.queryIdx].pt  for m in good], dtype=np.float32)
    pts_right = np.array([kp_right[m.trainIdx].pt for m in good], dtype=np.float32)
    return kp_left, kp_right, good, pts_left, pts_right


def draw_matches_img(img_left, img_right, kp_left, kp_right, matches, n=80):
    return cv2.drawMatches(
        img_left, kp_left, img_right, kp_right, matches[:n], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# -- Step 3: Fundamental matrix --

def estimate_fundamental(pts_left, pts_right):
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        raise RuntimeError("Fundamental matrix estimation failed.")
    inliers = mask.ravel().astype(bool)
    return F, pts_left[inliers], pts_right[inliers], inliers


# -- Step 4: Rectification --

def rectify_pair(left, right, F, pts_left, pts_right):
    h, w = left.shape[:2]
    ok, H_left, H_right = cv2.stereoRectifyUncalibrated(
        pts_left, pts_right, F, imgSize=(w, h))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed.")
    rect_left  = cv2.warpPerspective(left,  H_left,  (w, h))
    rect_right = cv2.warpPerspective(right, H_right, (w, h))
    return rect_left, rect_right, H_left, H_right


def make_valid_mask(shape, H_left, H_right, erosion_px=5):
    """Returns a boolean mask of pixels that are valid in both rectified images."""
    h, w = shape[:2]
    ones = np.ones((h, w), dtype=np.uint8) * 255
    mask_left  = cv2.warpPerspective(ones, H_left,  (w, h)) > 127
    mask_right = cv2.warpPerspective(ones, H_right, (w, h)) > 127
    combined   = (mask_left & mask_right).astype(np.uint8) * 255
    if erosion_px > 0:
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_RECT, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        combined = cv2.erode(combined, kernel)
    return combined.astype(bool)


def draw_epipolar_lines(img_left, img_right, num_lines=12):
    h          = img_left.shape[0]
    step       = max(1, h // (num_lines + 1))
    vis_left   = img_left.copy()
    vis_right  = img_right.copy()
    rng        = np.random.default_rng(42)
    for y in range(step, h, step):
        color = tuple(int(c) for c in rng.integers(80, 230, 3))
        cv2.line(vis_left,  (0, y), (img_left.shape[1]  - 1, y), color, 1)
        cv2.line(vis_right, (0, y), (img_right.shape[1] - 1, y), color, 1)
    return vis_left, vis_right


# -- Step 5: Disparity (SGBM + WLS, same implementation as ex3) --

def _create_sgbm(num_disparities):
    block_size = 5
    num_disp   = int(np.ceil(num_disparities / 16.0) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity_sgbm(gray_left, gray_right, num_disparities):
    num_disp   = int(np.ceil(num_disparities / 16.0) * 16)
    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity[disparity < 0] = np.nan
    return disparity


def compute_disparity_wls(gray_left, gray_right, num_disparities):
    left_matcher  = _create_sgbm(num_disparities)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    disp_left_raw  = left_matcher.compute(gray_left,  gray_right)
    disp_right_raw = right_matcher.compute(gray_right, gray_left)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    filtered   = wls_filter.filter(disp_left_raw, gray_left,
                                   disparity_map_right=disp_right_raw)
    confidence = wls_filter.getConfidenceMap()

    disparity = filtered.astype(np.float32) / 16.0
    disparity[disparity < 0] = np.nan
    return disparity, confidence


def colorize_disparity(disparity):
    vis = np.nan_to_num(disparity, nan=0.0)
    vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(vis), cv2.COLORMAP_PLASMA)


# -- Step 6: Depth map --

def disparity_to_depth(disparity, focal_px, baseline_m):
    depth = np.full_like(disparity, np.nan)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = focal_px * baseline_m / disparity[valid]
    return depth


def disparity_to_relative_depth(disparity):
    depth = np.full_like(disparity, np.nan)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = 1.0 / disparity[valid]
    return depth


def colorize_depth(depth):
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin = float(np.percentile(finite, 5))
    vmax = MAX_DEPTH_CM if MAX_DEPTH_CM is not None else float(np.percentile(finite, 95))
    vis  = np.clip(depth, vmin, vmax)
    vis  = np.nan_to_num(vis, nan=vmin)
    vis  = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(vis), cv2.COLORMAP_VIRIDIS)


# -- Visualisation --

def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _show_image(ax, img, title):
    ax.imshow(_bgr_to_rgb(img))
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def _show_map(ax, fig, data, cmap, title, colorbar_label,
             vmin_pct=2, vmax_pct=98, vmax=None):
    finite        = data[np.isfinite(data)]
    vmin          = float(np.percentile(finite, vmin_pct)) if finite.size else 0.0
    vmax          = vmax if vmax is not None else (
        float(np.percentile(finite, vmax_pct)) if finite.size else 1.0)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def _figsize_for_portrait(img_shape, num_cols, col_width=6.0):
    aspect = img_shape[0] / img_shape[1]
    return col_width * num_cols, col_width * aspect


def plot_rectification(rect_left_epi, rect_right_epi, inlier_matches_img,
                       n_inliers, n_matches, img_shape):
    fig_width, fig_height = _figsize_for_portrait(img_shape, num_cols=4)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.15, wspace=0.06)

    _show_image(fig.add_subplot(gs[0, 0]), rect_left_epi,
                "Rectified left (epipolar lines)")
    _show_image(fig.add_subplot(gs[0, 1]), rect_right_epi,
                "Rectified right (epipolar lines)")
    _show_image(fig.add_subplot(gs[0, 2:4]), inlier_matches_img,
                f"RANSAC inliers (top 50 of {n_inliers} / {n_matches})")
    return fig


def plot_stereo_results(sgbm_disparity, wls_disparity, depth, confidence,
                        depth_label, img_shape):
    fig_width, fig_height = _figsize_for_portrait(img_shape, num_cols=4)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.15, wspace=0.06)

    depth_unit = "cm" if BASELINE_CM else "rel"
    _show_map(fig.add_subplot(gs[0, 0]), fig,
              sgbm_disparity, "plasma", "Disparity - SGBM", "px")
    _show_map(fig.add_subplot(gs[0, 1]), fig,
              wls_disparity, "plasma", "Disparity - SGBM+WLS", "px")
    _show_map(fig.add_subplot(gs[0, 2]), fig,
              depth, "viridis", f"Depth map ({depth_label})",
              depth_unit, vmin_pct=5, vmax_pct=95, vmax=MAX_DEPTH_CM)
    _show_map(fig.add_subplot(gs[0, 3]), fig,
              confidence, "gray", "WLS confidence", "")
    return fig


def save_figure(fig, name):
    path = DATA / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# -- Main --

def main():
    print("=== Step 1: Loading images and calibration ===")
    left, right = load_images()
    K, _        = load_calibration()
    focal_px    = float(K[0, 0])
    print(f"  Image size   : {left.shape[1]} x {left.shape[0]}")
    print(f"  Focal length : {focal_px:.2f} px")
    if BASELINE_CM is not None:
        print(f"  Baseline     : {BASELINE_CM} cm  (metric depth enabled)")
    else:
        print("  Baseline     : not set  (relative depth only)")

    gray_left  = to_grayscale(left)
    gray_right = to_grayscale(right)

    print("\n=== Step 2: Detecting and matching features ===")
    kp_left, kp_right, good_matches, pts_left, pts_right = match_features(
        gray_left, gray_right)
    print(f"  Keypoints left / right  : {len(kp_left)} / {len(kp_right)}")
    print(f"  Matches (Lowe's ratio)  : {len(good_matches)}")

    print("\n=== Step 3: Estimating fundamental matrix ===")
    F, inliers_left, inliers_right, inlier_mask = estimate_fundamental(
        pts_left, pts_right)
    n_inliers = int(inlier_mask.sum())
    print(f"  F matrix               : {F.shape}")
    print(f"  Inliers after RANSAC   : {n_inliers} / {len(good_matches)}")
    inlier_matches     = [m for m, keep in zip(good_matches, inlier_mask) if keep]
    inlier_matches_img = draw_matches_img(
        left, right, kp_left, kp_right, inlier_matches, n=50)

    print("\n=== Step 4: Rectifying image pair ===")
    rect_left, rect_right, H_left, H_right = rectify_pair(
        left, right, F, inliers_left, inliers_right)
    valid_mask              = make_valid_mask(left.shape, H_left, H_right)
    rect_left_epi, rect_right_epi = draw_epipolar_lines(rect_left, rect_right)
    print("  Rectification complete.")
    cv2.imwrite(str(DATA / "ex4_rectified_left.png"),  rect_left)
    cv2.imwrite(str(DATA / "ex4_rectified_right.png"), rect_right)

    gray_rect_left  = to_grayscale(rect_left)
    gray_rect_right = to_grayscale(rect_right)

    print("\n=== Step 5: Computing disparity ===")
    sgbm_disparity = compute_disparity_sgbm(gray_rect_left, gray_rect_right, NDISP)
    sgbm_disparity[~valid_mask] = np.nan
    valid_pct = 100 * np.sum(np.isfinite(sgbm_disparity)) / sgbm_disparity.size
    print(f"  SGBM valid pixels  : {valid_pct:.1f}%")
    cv2.imwrite(str(DATA / "ex4_disparity_sgbm.png"),
                colorize_disparity(sgbm_disparity))

    wls_disparity, confidence = compute_disparity_wls(
        gray_rect_left, gray_rect_right, NDISP)
    wls_disparity[~valid_mask] = np.nan
    valid_pct = 100 * np.sum(np.isfinite(wls_disparity)) / wls_disparity.size
    print(f"  WLS  valid pixels  : {valid_pct:.1f}%")
    cv2.imwrite(str(DATA / "ex4_disparity_wls.png"),
                colorize_disparity(wls_disparity))

    print("\n=== Step 6: Computing depth map ===")
    if BASELINE_CM is not None:
        depth       = disparity_to_depth(wls_disparity, focal_px, BASELINE_CM)
        depth_label = f"metric, B={BASELINE_CM} cm"
    else:
        depth       = disparity_to_relative_depth(wls_disparity)
        depth_label = "relative (1/d)"
    print(f"  Depth type : {depth_label}")
    cv2.imwrite(str(DATA / "ex4_depth.png"), colorize_depth(depth))
    print(f"  Saved -> {DATA / 'ex4_depth.png'}")

    print("\n=== Saving figures ===")
    save_figure(
        plot_rectification(rect_left_epi, rect_right_epi, inlier_matches_img,
                           n_inliers, len(good_matches), left.shape),
        "ex4_rectification.png",
    )
    save_figure(
        plot_stereo_results(sgbm_disparity, wls_disparity, depth, confidence,
                            depth_label, left.shape),
        "ex4_stereo.png",
    )

    print("\nDone. All outputs in data/")


if __name__ == "__main__":
    main()
