# from ultralytics import YOLO
# import cv2

# # loading an model
# model = YOLO('yolov8n.pt')
# # starting an webcam
# cap = cv2.VideoCapture(0)

# # Infinite loop for continusly runing the code till programer stop
# while True :
#     ret, frame = cap.read()
#     if not ret:
#         break
#     r = model(frame)[0]
#     annomated_frame = r.plot()
#     annomated_frame = cv2.resize(annomated_frame, (1040, 720))
#     # count the person
#     count = sum(int(b.cls[0])==0 and float(b.conf[0])>0.7 for b in r.boxes)
#     cv2.rectangle(annomated_frame, (0, 0), (450, 60), (0, 0, 0), 2)
#     cv2.putText(annomated_frame, f'Person count: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.imshow('Person Detected', annomated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     frame = r.plot()
# cap.release()
# cv2.destroyAllWindows()





# number1 = 10
# number2 = 20
# print(f"before swap: number1 = {number1}, number2 = {number2}")
# temp = number1
# number1 = number2
# number2 = temp
# print(f"after swap: number1 = {number1}, number2 = {number2}")

#printing area of circle
radius = 5
print(f"Area of circle {radius} is: {3.14 * radius * radius }")