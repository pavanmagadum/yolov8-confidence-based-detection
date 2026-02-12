from ultralytics import YOLO
import cv2, os, requests
from datetime import datetime

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

os.makedirs('snapshots', exist_ok=True)
last_save = 0
cv2.namedWindow('Person + Phone Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Person + Phone Detection', 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]

    person_count = 0
    phone_detected = False

    if r.boxes is not None:
        for b in r.boxes:
            class_id = int(b.cls[0])
            confidence = float(b.conf[0])

            if class_id == 0 and confidence > 0.7:
                person_count += 1

            if class_id == 67 and confidence > 0.7:
                phone_detected = True

    frame = r.plot()

    cv2.putText(frame, f'Person Count: {person_count}',(20, 80),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2)

    # Capture snapshot only if person + phone detected
    current_time = datetime.now().timestamp()
    if person_count >= 1 and phone_detected and (current_time - last_save) >= 3:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)
        cv2.putText(frame,f"{timestamp} | Person + Phone Detected",(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)

        cv2.imwrite(f'snapshots/snapshot_{timestamp}.jpg', frame)
        last_save = current_time
    
    cv2.imshow('Person + Phone Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord
        break

cap.release()
cv2.destroyAllWindows()
