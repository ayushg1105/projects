import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    if not ret:
        break

    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- BLUE MASK ---
    lb_limit = np.array([90, 50, 50]) 
    ub_limit = np.array([130, 255, 255]) 
    b_mask = cv2.inRange(into_hsv, lb_limit, ub_limit)
    blue = cv2.bitwise_and(frame, frame, mask=b_mask)
    cv2.imshow('Blue Detector', blue)

    # --- RED MASK (Wraps around HSV) ---
    lr1 = np.array([0, 70, 50])
    ur1 = np.array([10, 255, 255])
    lr2 = np.array([170, 70, 50])
    ur2 = np.array([180, 255, 255])
    r_mask1 = cv2.inRange(into_hsv, lr1, ur1)
    r_mask2 = cv2.inRange(into_hsv, lr2, ur2)
    r_mask = r_mask1 + r_mask2
    red = cv2.bitwise_and(frame, frame, mask=r_mask)
    cv2.imshow('Red Detector', red)
    
    # --- GREEN MASK ---
    lg_limit = np.array([35, 50, 50]) 
    ug_limit = np.array([85, 255, 255]) 
    g_mask = cv2.inRange(into_hsv, lg_limit, ug_limit)
    green = cv2.bitwise_and(frame, frame, mask=g_mask)
    cv2.imshow('Green Detector', green)

    cv2.imshow('Original', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()