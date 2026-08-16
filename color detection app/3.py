import cv2
import numpy as np
import pandas as pd

# Load the color dataset
index = ["color", "color_name", "hex", "R", "G", "B"]
try:
    csv = pd.read_csv('colors.csv', names=index, header=None)
except FileNotFoundError:
    print("Error: colors.csv not found in the directory.")
    exit()

def getColorName(R, G, B):
    # Vectorized distance calculation for high-speed real-time performance
    d = abs(csv['R'] - R) + abs(csv['G'] - G) + abs(csv['B'] - B)
    idx = d.idxmin()
    return csv.loc[idx, 'color_name']

def nothing(x):
    pass

# Setup webcam and window
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Advanced Color Tracker")

# Create a Trackbar to switch between modes
# 0 = All Colors (Dataset), 1 = Red, 2 = Green, 3 = Blue, 4 = Off
cv2.createTrackbar("Mode", "Advanced Color Tracker", 0, 4, nothing)

# Define generic mask bounds for 'All Colors' (filtering out grayscale/black/white)
low_color = np.array([0, 50, 50])
high_color = np.array([180, 255, 255])
kernal = np.ones((5, 5), "uint8")

while True:
    _, frame = webcam.read()
    if frame is None:
        break
        
    frame = cv2.flip(frame, 1) # Mirror for natural feel
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mode = cv2.getTrackbarPos("Mode", "Advanced Color Tracker")

    if mode == 4:
        # Off mode: just show frame
        cv2.putText(frame, "Mode: OFF", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Advanced Color Tracker", frame)
    
    elif mode == 0:
        # All Colors Mode (Dataset Matching)
        cv2.putText(frame, "Mode: ALL COLORS (Dataset)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Mask out non-colorful things
        mask = cv2.inRange(hsvFrame, low_color, high_color)
        mask = cv2.dilate(mask, kernal)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000: # Slightly larger threshold to reduce noise
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get the color of the center pixel of the bounding box
                cx = x + w // 2
                cy = y + h // 2
                b, g, r = frame[cy, cx]
                
                color_name = getColorName(int(r), int(g), int(b))
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (int(b), int(g), int(r)), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (int(b), int(g), int(r)), 2)

    elif mode in [1, 2, 3]:
        mask = None
        color_bgr = (0, 0, 0)
        label = ""

        if mode == 1:
            cv2.putText(frame, "Mode: SPECIFIC (Red)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mask1 = cv2.inRange(hsvFrame, np.array([0, 70, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsvFrame, np.array([170, 70, 50]), np.array([180, 255, 255]))
            mask = mask1 + mask2
            color_bgr = (0, 0, 255)
            label = "Red"
        elif mode == 2:
            cv2.putText(frame, "Mode: SPECIFIC (Green)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mask = cv2.inRange(hsvFrame, np.array([35, 50, 50]), np.array([85, 255, 255]))
            color_bgr = (0, 255, 0)
            label = "Green"
        elif mode == 3:
            cv2.putText(frame, "Mode: SPECIFIC (Blue)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            mask = cv2.inRange(hsvFrame, np.array([90, 50, 50]), np.array([130, 255, 255]))
            color_bgr = (255, 0, 0)
            label = "Blue"

        mask = cv2.dilate(mask, kernal)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    cv2.imshow("Advanced Color Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
        break

webcam.release()
cv2.destroyAllWindows()
