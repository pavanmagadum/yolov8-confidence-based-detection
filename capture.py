from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

conf_threshold = 0.70  # confidence threshold
person_saved = False  # ✅ move OUTSIDE while

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, conf=0.7)[0]
    person_count = 0

    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])

        # ✅ Save person image only once if conf > 0.7
        if cls_id == 0 and conf > conf_threshold and not person_saved:
            cv2.imwrite("person_detected.jpg", frame)
            person_saved = True
            print("Person image captured")

        # ✅ Count person with conf > 0.7
        if cls_id == 0 and conf > conf_threshold:
            person_count += 1

    annotated_frame = r.plot()
    if person_saved:
        cv2.putText(
        annotated_frame,
        "Person Captured",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f'Person Count (conf > 0.7): {person_count}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
