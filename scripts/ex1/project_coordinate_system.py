import cv2 as cv
import numpy as np

# --- Load calibration ---
data = np.load("calibration.npz")
mtx = data["camera_matrix"]
dist = data["dist_coeffs"]

# --- Chessboard config ---
CHESSBOARD_SIZE = (9, 6)

# prepare object points (same as calibration)
ccols, crows = CHESSBOARD_SIZE
objp = np.zeros((crows * ccols, 3), np.float32)
objp[:, :2] = np.mgrid[0:ccols, 0:crows].T.reshape(-1, 2)

# axis: 3 units long in each direction
axis = np.float32([
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, -3]
]).reshape(-1, 3)

id = 0
cap = cv.VideoCapture(id)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        # refine corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # solve PnP
        success, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        if success:
            # project 3D axis points
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

            corner = tuple(corners2[0].ravel().astype(int))

            # draw axes
            cv.line(frame, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # X (red)
            cv.line(frame, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Y (green)
            cv.line(frame, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Z (blue)

        cv.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, found)

    cv.imshow("Pose Estimation", frame)

    if cv.waitKey(1) == ord('q'):  # ESC
        break

cap.release()
cv.destroyAllWindows()