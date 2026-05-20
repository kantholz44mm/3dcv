import cv2
from datetime import datetime
import os
import glob

# ---- camera settings ----
CAMERA_ID  = 0
WIDTH      = 1920
HEIGHT     = 1080
FPS        = 10     # lower FPS lets the sensor expose longer → better quality

# Connect to the webcam
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          FPS)

actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Requested : {WIDTH}×{HEIGHT} @ {FPS} fps")
print(f"Negotiated: {actual_w}×{actual_h} @ {actual_fps} fps")

# Create a window to display the webcam feed
cv2.namedWindow("Webcam")

# Main loop to capture images
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Show the webcam feed
    cv2.imshow("Webcam", frame)

    # Handle key presses
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('s'):
        suffix = str(datetime.now())
        filename = f'captured_images/{suffix}.png'
        cv2.imwrite(filename, frame)
    elif key == ord('d'):
        files = glob.glob('captured_images/*')
        for file in files:
            os.remove(file)


# Release resources
cap.release()
cv2.destroyAllWindows()