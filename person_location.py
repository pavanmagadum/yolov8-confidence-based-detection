from ultralytics import YOLO
import cv2, os,requests
from datetime import datetime

model, cap = YOLO('yolov8n.pt'), cv2.VideoCapture(0)
os.makedirs('snapshots', exist_ok=True)
last_save=0
confidence_threshold = 0.7
cv2.namedWindow('Person + Phone Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Person + Phone Detection', 1280, 720)

def get_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
        return {
            'city': data.get('city', 'N/A'),
            'region': data.get('region', 'N/A'),
            'country': data.get('country', 'N/A'),
            'lat': data.get('loc', 'N/A').split(',')[0],
            'lon': data.get('loc', 'N/A').split(',')[1]
        }
    except:
        pass
    return {'city': 'N/A', 'region': 'N/A', 'country': 'N/A', 'lat': 'N/A', 'lon': 'N/A'}
location = get_location()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]
    person_count = sum(int(b.cls[0]) == 0  for b in r.boxes)  # person class_id=0
    frame = r.plot()
    cv2.putText(frame, f'Person Count: {person_count}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    location = get_location()
    cv2.putText(frame, f"Location: {location}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if person_count > 2 and (datetime.now().timestamp() - last_save) >= 10:  # Save snapshot every 3 seconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.rectangle(frame,(0,0),(450,60),(255,0,0),2)
        cv2.putText(frame,f"{timestamp} | Persons: {person_count}",(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(f'snapshots/snapshot_{timestamp}.jpg', frame)
        last_save = datetime.now().timestamp()

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()