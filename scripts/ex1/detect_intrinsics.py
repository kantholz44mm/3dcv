import numpy as np
import cv2 as cv
import glob

CHESSBOARD_SIZE = (9, 6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points (0,0,0), (1,0,0), ...
ccols, crows = CHESSBOARD_SIZE
objp = np.zeros((crows * ccols, 3), np.float32)
objp[:, :2] = np.mgrid[0:ccols, 0:crows].T.reshape(-1, 2)

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob('captured_images/*.png')

for img_path in images:
    print(f'checking: {img_path}')

    raw = cv.imread(img_path)
    gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        print('detected chessboard')

        # refine corners
        corners_subpix = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_subpix)

        # visualize
        cv.drawChessboardCorners(raw, CHESSBOARD_SIZE, corners_subpix, ret)
        cv.imshow('img', raw)
        cv.waitKey(500)

    else:
        print("Failed to find chessboard pattern")

cv.destroyAllWindows()

# --- IMPORTANT: calibrate once, after all images ---
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    np.savez(
        'calibration.npz',
        camera_matrix=mtx,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs
    )

    print(f"Calibration saved to 'calibration.npz'")
    print(f"Reprojection error: {ret}")
else:
    print("No valid chessboard detections found.")