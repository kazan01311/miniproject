from ultralytics import YOLO
import torch
import cv2

print(torch.__version__)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n.pt')
print(model.names)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[5, 7], device='cuda')  # traffic
    # results = model(frame, classes=[15, 16], device='cuda')  # cat and dog

    detect_frame = results[0].plot()

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        # print('conficences : %f' % (conf))
        print('confidences : %f | box : (x1 : %f, y1 : %f, x2 : %f, y2 : %f)' % (conf, x1, y1, x2, y2))

    cv2.imshow('frame', detect_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
