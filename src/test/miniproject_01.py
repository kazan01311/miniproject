from pandas import wide_to_long
from ultralytics import YOLO
import torch
import cv2
import math
import random


speed = 0

print(torch.__version__)

bumper_img = cv2.imread('../img/front_bumper.png', cv2.IMREAD_UNCHANGED)


# 자동차 범퍼 앞면 포인터
def pointer(fream):
    height, width = frame.shape[:2]
    bumper_x = width // 2
    bumper_y = height - 100
    cv2.circle(frame, (bumper_x, bumper_y), 5, (0, 0, 255), -1)

    return bumper_x, bumper_y


# 감지한 자동차 뒷면 포인터
def target_pointer(frame, x1, y1, x2, y2):
    height, width = frame.shape[:2]
    cx = int((x1 + x2) / 2)  # 박스 가로 중앙
    cy = int(y2)  # 박스 맨 아래
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return cx, cy


# 자동차 범퍼 이미지 (앞)
def front_bumper(frame):
    height, width = frame.shape[:2]
    bumper_h, bumper_w = bumper_img.shape[:2]
    x_offset = width // 2 - bumper_w // 2
    y_offset = height - bumper_h
    frame[y_offset: y_offset + bumper_h, x_offset: x_offset + bumper_w] = bumper_img[:, :, :3]


def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def line(frame, x1, y1, x2, y2):
    cv2.line(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n.pt')
print(model.names)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    m_x, m_y = pointer(frame)
    front_bumper(frame)

    results = model(frame, classes=[5, 7], device='cuda', verbose=False)  # traffic
    # results = model(frame, classes=[15, 16], device='cuda')  # cat and dog

    detect_frame = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    confidences = results[0].boxes.conf.cpu().numpy()

    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        # print('conficences : %f' % (conf))
        # print('confidences : %f | box : (x1 : %f, y1 : %f, x2 : %f, y2 : %f)' % (conf, x1, y1, x2, y2))
        t_x, t_y = target_pointer(frame, x1, y1, x2, y2)

        distance = euclidean_distance(m_x, m_y, t_x, t_y)
        line(frame, m_x, m_y, t_x, t_y)

        cv2.imshow('detect_frame', detect_frame)
    cv2.imshow('frame', frame)
    cv2.imshow('detect_frame', detect_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
