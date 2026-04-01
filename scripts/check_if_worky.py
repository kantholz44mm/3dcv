import cv2

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
    match key:
        case 113: # ord('q') = 113 (quit)
            break

# Release resources
cap.release()
cv2.destroyAllWindows()