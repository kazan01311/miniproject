from ultralytics import YOLO
import mss
import airsim
import time
import numpy as np
import cv2

model = YOLO('yolov8n.pt',)

client = airsim.CarClient()
client.confirmConnection()  # 연결 확인
client.enableApiControl(True)  # Python API 제어 활성화
# client.enableApiControl(False)  # Python API 제어 비활성화
car_controls = airsim.CarControls()


# 전진/후진 + 조향 제어 함수
def drive(throttle=0.0, steering=0.0, duration=1.0):
    """
    throttle : 속도 제어, -1.0 ~ 1.0
               양수: 전진, 음수: 후진
    steering : 조향, -1.0 ~ 1.0
               음수: 좌회전, 양수: 우회전
    duration : 해당 명령 유지 시간 (초)
    """
    car_controls.throttle = max(min(throttle, 1.0), -1.0)  # -1~1 범위 제한
    car_controls.steering = max(min(steering, 1.0), -1.0)

    # 브레이크는 throttle이 0일 때만 작동
    car_controls.brake = 1.0 if throttle == 0 else 0.0

    client.setCarControls(car_controls)
    time.sleep(duration)

    # 지속적으로 제어하지 않을 경우 정지
    car_controls.throttle = 0
    car_controls.steering = 0
    car_controls.brake = 1
    client.setCarControls(car_controls)

# Main loop to continuously get images and perform detection
try:
    while True:

        # car_controls.throttle = 0.5  # 전진
        # car_controls.steering = 0.0  # 직진
        # client.setCarControls(car_controls)

        # 센서 감지 코드 에러
        # dist_data = client.getDistanceSensorData(distance_sensor_name="MyDistance")
        # print(dist_data.distance)

        # Get an image from a specific camera
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        # Process the image data
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # Convert the image to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Perform object detection with YOLO
        results = model(img_bgr, conf=0.5)

        # Get the detection results (e.g., bounding boxes, class names)
        # You can process these results further
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Draw the bounding box and label on the image
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_bgr, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

        # Display the processed image
        cv2.imshow("YOLO Detection", img_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        drive(throttle=0.5, steering=0.0, duration=1)

finally:
    # Release resources
    client.enableApiControl(False)
    cv2.destroyAllWindows()


# 사용 예시
# drive(throttle=0.5, steering=0.0, duration=2)  # 2초 전진
# drive(throttle=-0.5, steering=0.0, duration=1)  # 1초 후진
# drive(throttle=0.5, steering=0.5, duration=2)  # 2초 우회전 전진
# drive(throttle=0.5, steering=-0.5, duration=2)  # 2초 좌회전 전진

