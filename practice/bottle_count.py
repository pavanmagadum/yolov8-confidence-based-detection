from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame)[0]

    count = sum(int(b.cls[0]) == 39 and b.conf[0] > 0.7 for b in r.boxes)

    frame = r.plot()
    cv2.putText(frame, f"Bottles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Bottle Count", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
