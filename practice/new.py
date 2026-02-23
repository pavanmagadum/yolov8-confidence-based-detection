# from ultralytics import YOLO
# import cv2

# model = YOLO('best.pt')

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
    
#     results = model(frame)
    
#     annotated_frame = results[0].plot()

#     annotated_frame = cv2.resize(annotated_frame, (840, 580))
        
#     cv2.imshow('YOLO Object Detection', annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



from ultralytics import YOLO
import cv2, os
from datetime import datetime

model, cap = YOLO("yolov8n.pt"), cv2.VideoCapture(0)
os.makedirs("snapshots", exist_ok=True)
last_save = 0

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
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{timestamp} | Persons: {person_count}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
        print(f"✓ Saved snapshot")
        last_save = datetime.now().timestamp()
    
    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
