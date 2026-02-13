from ultralytics import YOLO
import cv2, os,requests
from datetime import datetime


FIREBASE_URL = 'https://crop-recommendation-3f8a5-default-rtdb.firebaseio.com/' # Replace with your Firebase URL
model, cap = YOLO('yolov8n.pt'), cv2.VideoCapture(0)
os.makedirs('snapshots', exist_ok=True)
last_save=0
last_firebase_update=0

def send_to_firebase(bottle_detected):
    try:
        data={
            'bottle_detected': 1 if bottle_detected else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'bottle_count': bottle_count
            }
        response = requests.put(f"{FIREBASE_URL}/bottle_detection.json", json=data)
        if response.status_code == 200:
            print(f"Firebase updated: bottle_detected = {1 if bottle_detected else 0}")
        else:
            print(f"Firebase error{response.status_code}")
    except Exception as e:
        print(f"Error sending to Firebase: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]
    bottle_count = sum(int(b.cls[0]) == 39  for b in r.boxes)  # bottle class_id=39
    frame = r.plot()
    cv2.putText(frame, f'Bottles: {bottle_count}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if bottle_count > 0 and (datetime.now().timestamp() - last_firebase_update) >= 5: 
        send_to_firebase(bottle_count > 0)
        last_firebase_update = datetime.now().timestamp()
    
    if bottle_count > 0 and (datetime.now().timestamp() - last_save) >= 50:  # Save snapshot every 3 seconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.rectangle(frame,(0,0),(450,60),(255,0,0),2)
        cv2.putText(frame,f"{timestamp} | Bottles: {bottle_count}",(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(f'snapshots/snapshot_{timestamp}.jpg', frame)
        print(f"Saved snapshot")
        last_save = datetime.now().timestamp()

    cv2.imshow('Bottle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
send_to_firebase(False)
cap.release()
cv2.destroyAllWindows()