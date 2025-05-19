import cv2
import numpy as np

# Define color ranges for red, yellow, and green in HSV
RED_LOWER = np.array([0, 120, 70])
RED_UPPER = np.array([10, 255, 255])
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])

def detect_traffic_signal(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect red color
    red_mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Detect yellow color
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Detect green color
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check for red signal
    for contour in red_contours:
        if cv2.contourArea(contour) > 500:  # Adjust the area threshold based on your setup
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "RED SIGNAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

    # Check for yellow signal
    for contour in yellow_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "YELLOW SIGNAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame

    # Check for green signal
    for contour in green_contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "GREEN SIGNAL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame

    return frame

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_traffic_signal(frame)
    cv2.imwrite("output.jpg", processed_frame)
    print("Frame saved to output.jpg")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
