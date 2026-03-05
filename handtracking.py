from ultralytics import YOLO
import cv2

# load pretrained YOLO model
model = YOLO("yolov8n.pt")

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # run detection
    results = model(frame)

    # draw boxes
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()