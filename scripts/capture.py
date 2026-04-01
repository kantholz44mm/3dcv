import cv2
from datetime import datetime
import os
import glob

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