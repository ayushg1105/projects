import cv2
import numpy as np
import tensorflow as tf
import time 

print("Loading optimized model...")
# Make sure to use the newly trained model!
model = tf.keras.models.load_model('face_mask_detector.h5') 
print("Model loaded successfully!")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

prev_frame_time = 0

print("Starting webcam... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
    gray_frame, 
    scaleFactor=1.05,  # Makes the algorithm scan more thoroughly
    minNeighbors=3,    # Lowers the strictness for confirming a face
    minSize=(50, 50)   # Allows it to detect faces slightly further away
)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # --- THE FIX: Convert OpenCV's BGR format to TensorFlow's RGB format ---
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Now resize and normalize the color-corrected face
        resized_face = cv2.resize(rgb_face, (128, 128))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 128, 128, 3))

        # Make the prediction
        prediction = model.predict(reshaped_face, verbose=0)[0][0]

        if prediction < 0.5:
            label = "Mask"
            color = (0, 255, 0)  # Green
            confidence = (1 - prediction) * 100 
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red
            confidence = prediction * 100

        # 1. Draw Bounding Box and Text
        label_text = f"{label}: {round(confidence, 1)}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. NEW: Draw Real-Time Dynamic Confidence Bar
        bar_x = x
        bar_y = y + h + 10  # Place it just below the bounding box
        bar_width = w
        bar_height = 10
        
        # Calculate how much of the bar to fill based on the percentage
        fill_width = int((confidence / 100) * bar_width)

        # Draw the background of the bar (Dark Gray)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Draw the filled portion of the bar (Matches the Red/Green color)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)


    # 3. Static HUD (Heads Up Display) in the corner
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Display FPS and system status
    cv2.putText(frame, f"Live FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Status: Monitoring...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Face Mask Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()