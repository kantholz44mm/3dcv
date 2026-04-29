import cv2
from datetime import datetime
import os
import glob
import numpy as np

# load our calibration data
data = np.load("calibration.npz")
mtx   = data["camera_matrix"]
dist  = data["dist_coeffs"]
rvecs = data["rvecs"]
tvecs = data["tvecs"]

print(mtx)
print(dist)
print(rvecs)
print(tvecs)

# Connect to the webcam
id = 0
cap = cv2.VideoCapture(id)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow("Webcam")

# Main loop to capture images
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    h, w = frame.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    frame_but_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Show the webcam feed
    cv2.imshow("Webcam", frame)
    cv2.imshow("Webcam but undistorted", frame_but_undistorted)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()