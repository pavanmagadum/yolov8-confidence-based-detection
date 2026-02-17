from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import requests
from datetime import datetime

# Firebase URL
FIREBASE_URL = 'https://crop-recommendation-3f8a5-default-rtdb.firebaseio.com/'

# Initialize YOLO model and OCR reader
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

last_firebase_update = 0
last_save = 0

def send_to_firebase(plate_detected, plate_text, plate_count):
    try:
        data = {
            'plate_detected': 1 if plate_detected else 0,
            'plate_text': plate_text,
            'plate_count': plate_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        response = requests.put(f"{FIREBASE_URL}/plate_detection.json", json=data)
        if response.status_code == 200:
            print("Firebase updated successfully")
        else:
            print("Firebase error:", response.status_code)
    except Exception as e:
        print("Error sending to Firebase:", e)


def extract_plate_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)

        plates = []
        for (bbox, text, conf) in results:
            if conf > 0.3 and len(text) >= 4:
                text = ''.join(c for c in text if c.isalnum() or c == ' ')
                plates.append((text.upper(), conf))

        return plates
    except:
        return []


frame_count = 0
plates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 480))

    if frame_count % 10 == 0:
        plates = extract_plate_text(frame)

    y_pos = 40
    for text, conf in plates:
        accuracy = int(conf * 100)
        cv2.putText(frame, f"{text} ({accuracy}%)", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 35

    plate_count = len(plates)

    cv2.putText(frame, f"Plates: {plate_count}", 
                (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Firebase update every 5 seconds if plate detected
    if plate_count > 0 and (datetime.now().timestamp() - last_firebase_update) >= 5:
        send_to_firebase(True, plates[0][0], plate_count)
        last_firebase_update = datetime.now().timestamp()

    # Save snapshot every 50 seconds if detected
    if plate_count > 0 and (datetime.now().timestamp() - last_save) >= 50:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'snapshots/plate_{timestamp}.jpg', frame)
        print("Saved snapshot")
        last_save = datetime.now().timestamp()

    cv2.imshow("License Plate Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

send_to_firebase(False, "", 0)
cap.release()
cv2.destroyAllWindows()
