from ultralytics import YOLO
import mss
import airsim
import time
import numpy as np
import cv2
import keyboard


# def drive(throttle=0.0, steering=0.0, duration=1.0):
#     car_controls.throttle = max(min(throttle, 1.0), -1.0)
#     car_controls.steering = max(min(steering, 1.0), -1.0)
#     car_controls.brake = 1.0 if throttle == 0 else 0.0
#     client.setCarControls(car_controls)
#     time.sleep(duration)
#     car_controls.throttle = 0
#     car_controls.steering = 0
#     car_controls.brake = 1
#    C

# def avoid(bounding_box_size, x1, y1, x2, y2):
# #     if bounding_box_size > 1500:
# #         if x1 >= 128:
# #             car_controls.throttle = 0.5
# #             car_controls.steering = 1.0
# #             client.setCarControls(car_controls)
# #
# #         elif x2 <= 128:
# #             car_controls.throttle = 0.5
# #             car_controls.steering = -1.0
# #             client.setCarControls(car_controls)
# #
# #         else:
# #             print('이상없음')

# 장애물 회피 함수 (좌,우)
def avoid(bounding_box_size, x1, x2):
    if bounding_box_size > 2000:
        center_x = (x1 + x2) / 2
        screen_center_x = 128

        # 오른쪽에 장애물 → 왼쪽으로 회피
        if center_x > screen_center_x:
            print('오른쪽 자동차 발견! 왼쪽으로 회피합니다.')
            car_controls.throttle = 0.5
            car_controls.steering = -0.5
            client.setCarControls(car_controls)
            time.sleep(1)  # 회피 지속 시간

            # 보정 (반대방향)
            car_controls.steering = 0.5
            client.setCarControls(car_controls)
            time.sleep(1)

            # 직선 복귀
            car_controls.steering = 0.0
            client.setCarControls(car_controls)

        # 왼쪽에 장애물 → 오른쪽으로 회피
        elif center_x < screen_center_x:
            print('왼쪽 자동차 발견! 오른쪽으로 회피합니다.')
            car_controls.throttle = 0.5
            car_controls.steering = 0.5
            client.setCarControls(car_controls)
            time.sleep(1)

            # 보정 (반대방향)
            car_controls.steering = -0.5
            client.setCarControls(car_controls)
            time.sleep(1)

            # 직선 복귀
            car_controls.steering = 0.0
            client.setCarControls(car_controls)

        else:
            print('.')

    # 장애물이 멀리 있으면 기본값(직진)을 반환합니다.
    return 0.5, 0.0


def main():
    try:
        while True:

            # 1. 루프 시작 마다 입력값 초기화
            car_controls.throttle = 0.0
            car_controls.steering = 0.0
            car_controls.brake = 0.0


            # 2. 키보드 입력 상태에 따라 제어값을 설정합니다.
            if keyboard.is_pressed('up'):
                car_controls.throttle = 0.5
            if keyboard.is_pressed('down'):
                car_controls.is_manual_gear = True  # 수동 기어 모드
                car_controls.manual_gear = -1  # 후진 기어
            else:
                car_controls.is_manual_gear = False
                car_controls.throttle = 0.5
            if keyboard.is_pressed('left'):
                car_controls.steering = -1.0
            if keyboard.is_pressed('right'):
                car_controls.steering = 1.0

            airsim.time.sleep(0.01)

            # 3. 시뮬레이션 속의 자동차 0번 인덱스 카메라 반환
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            response = responses[0]
            # width = response.width
            # height = response.height
            # screen_size = width * height
            # print("screen_size: {} ".format(screen_size))
            # print("width: {}, height: {}".format(width, height))

            # YOLO Detection screen info

            '''
            width = 256, height: 144
            '''

            # 4. 버퍼 데이터를 이미지 변환 후, RGB 타입을 BGR 로 변경
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # YOLO 자동차 객체 탐지 (In Simulation)
            results = model(img_rgb, conf=0.5, classes=[2])


            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    print("x1: {}, y1: {}, x2: {}, y2: {},".format(x1, y1, x2, y2))
                    bounding_box_width = x2 - x1
                    bounding_box_height = y2 - y1
                    bounding_box_size = bounding_box_width * bounding_box_height
                    print("bounding box size: {}".format(bounding_box_size))

                    if bounding_box_size > 2000:
                        print('Avoid !!!')
                        car_controls.throttle, car_controls.steering = avoid(bounding_box_size, x1, x2)
                        client.setCarControls(car_controls)

                    cv2.putText(img_bgr, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

            client.setCarControls(car_controls)
            cv2.imshow("YOLO Detection", img_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # drive(throttle=0.5, steering=0.0, duration=1)

    finally:
        # Disable API control only once at the end of the program
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO('yolov8m.pt')
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    client.setCarControls(car_controls)

    main()
