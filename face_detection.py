from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')

cap=cv2.VideoCapture(0)
cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    'YOLOv8 Detection',
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret,frame=cap.read()

    results=model(frame)

    annotated_frame=results[0].plot()

    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()