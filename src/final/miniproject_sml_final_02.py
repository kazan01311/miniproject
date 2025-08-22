from ultralytics import YOLO
import airsim
import time
import numpy as np
import cv2
import keyboard

# 전역 변수
model = YOLO('yolov8m.pt')
client = airsim.CarClient()
car_controls = airsim.CarControls()


# AirSim 서버와 클라이언트 연결
def init_airsim_client():
    client.confirmConnection()
    client.enableApiControl(True)
    client.setCarControls(car_controls)
    print("AirSim 클라이언트 연결완료.")


# 방향키 조작 앞, 뒤, 좌, 우
def process_keyboard_input():
    # Max 1.0 Min -1.0
    # throttle : 앞, 뒤  가속도, steering : 좌, 우 조향각 조정
    car_controls.throttle = 0.0
    car_controls.steering = 0.0
    car_controls.is_manual_gear = False

    if keyboard.is_pressed('up'):
        car_controls.throttle = 0.8
    elif keyboard.is_pressed('down'):
        car_controls.is_manual_gear = True  # 수동 기어모드 황성화
        car_controls.manual_gear = -1  # 후진 기어 설정
        car_controls.throttle = 0.5
    else:
        car_controls.throttle = 0.65
        car_controls.is_manual_gear = False  # 수동 기어모드 비활성화

    if keyboard.is_pressed('left'):  # 조향각 값이 음수면 왼쪽
        car_controls.steering = -0.5
    elif keyboard.is_pressed('right'):  # 조향각 값이 양수면 오른쪽
        car_controls.steering = 0.5


# 인게임 자동차 전방 이미지 가져오기
def get_image_data():
    # 인게임 자동차 전방 이미지를 1차원 바이트배열로 반환하는 함수
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    response = responses[0]

    # 1차원 바이트 배열을 넘파이 3채널 배열로 변환 후 OpenCV 처리를 위해 RGB2BGR 변환
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


# 대형차 감지
def process_yolo_detection(img_bgr, detection_classes, slow_threshold, break_threshold):
    # YOLO 트럭과 버스만 탐지
    results = model(img_bgr, conf=0.5, classes=detection_classes, verbose=False)
    is_slow_needed = False
    is_break_needed = False

    # 탐지한 객체 바운스 박스 x1, y1, x2, y2 좌표 저장
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 탐지한 바운드 박스 x, y좌표를 이용해 박스의 크기를 계산
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            bounding_box_size = (x2 - x1) * (y2 - y1)

            # 출력
            print(f"Detected: {class_name}, Size: {bounding_box_size}")

            # 감지한 바운드 박스가 임계값 보다 크다면 감속주행 (1500)
            if bounding_box_size > slow_threshold:
                print(f'Warning: {class_name} is nearby.')
                is_slow_needed = True

            # 감지한 바운드 박스가 임계값 보다 크다면 정지 (3000)
            if bounding_box_size > break_threshold:
                print('Break!')
                is_break_needed = True

            # 박스 선과 이름 지정
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_bgr, f'{class_name} {confidence:.2f}', (x1, y1 - 10), 0, 0.5,
                        (255, 0, 0), 2)

    return img_bgr, is_slow_needed, is_break_needed


def main_loop():
    try:
        while True:
            # 방향키 함수
            process_keyboard_input()

            # 이미지 데이터 가져오기
            img_bgr = get_image_data()

            # YOLO 탐지 및 브레이크 필요 여부 확인
            detected_img, is_slow, is_break = process_yolo_detection(img_bgr, [2, 5, 7], 1500, 3000)

            # 자동 제어 (속도조절)
            if is_break:
                car_controls.throttle = 0.0
                cv2.putText(img_bgr, 'Break!', (116, 72), 0, 0.5,
                            (0, 0, 255), 2)
            elif is_slow: # 'is_break'가 False일 경우에만 'is_slow'를 확인
                car_controls.throttle = 0.3
                cv2.putText(img_bgr, 'Slow', (116, 72), 0, 0.5,
                            (0, 0, 255), 2)
            else: # is_slow와 is_break 모두 False일 때 기본 스로틀로 복귀
                car_controls.throttle = 0.5

            # 차량 제어 적용 및 화면 표시
            client.setCarControls(car_controls)
            cv2.imshow("YOLO Detection", detected_img)

            # imshow 함수 닫기
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # 인게임에서 't'를 누를시 자동차 위치 초기화
            if keyboard.is_pressed('t'):
                client.reset()
                client.enableApiControl(True)

    finally:

        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    init_airsim_client()
    main_loop()
