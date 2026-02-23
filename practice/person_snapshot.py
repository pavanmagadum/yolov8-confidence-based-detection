from ultralytics import YOLO
import cv2, os, requests
from datetime import datetime

model, cap = YOLO("yolov8n.pt"), cv2.VideoCapture(0)
os.makedirs("snapshots", exist_ok=True)
last_save = 0

# Get location once at startup
def get_location():
    try:
        response = requests.get("https://ipapi.co/json/")
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'lat': data.get('latitude', 0),
                'lng': data.get('longitude', 0)
            }
    except:
        pass
    return {'city': 'Unknown', 'region': 'Unknown', 'country': 'Unknown', 'lat': 0, 'lng': 0}

location = get_location()
print(f"📍 Location: {location['city']}, {location['region']}, {location['country']}")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    r = model(frame)[0]
    person_count = sum(int(b.cls[0]) == 0 for b in r.boxes)
    frame = r.plot()
    cv2.putText(frame, f"Persons: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                 1, (0, 255, 0), 2)
    
    if person_count > 0 and (datetime.now().timestamp() - last_save) >= 3:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        
        location = get_location()
        
        # Draw black background for text
        cv2.rectangle(frame, (0, 0), (500, 90), (0, 0, 0), -1)
        
        # Add timestamp and person count
        cv2.putText(frame, f"{timestamp} | Persons: {person_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add location info
        cv2.putText(frame, f"Location: {location['city']}, {location['region']}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Coords: {location['lat']}, {location['lng']}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Save snapshot
        filename = f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save location details to text file
        with open(filename.replace('.jpg', '.txt'), 'w') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Persons Detected: {person_count}\n")
            f.write(f"City: {location['city']}\n")
            f.write(f"Region: {location['region']}\n")
            f.write(f"Country: {location['country']}\n")
            f.write(f"Latitude: {location['lat']}\n")
            f.write(f"Longitude: {location['lng']}\n")
            f.write(f"Google Maps: https://www.google.com/maps?q={location['lat']},{location['lng']}\n")
        
        print(f"✓ Saved snapshot with location: {location['city']}")
        last_save = datetime.now().timestamp()
    
    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
