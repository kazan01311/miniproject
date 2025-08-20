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

def avoid(bounding_box_size, x1, y1, x2, y2):
    if bounding_box_size > 1500:
        # 객체가 화면 중앙에 있는지 확인
        center_x = (x1 + x2) / 2
        screen_center_x = 128  # 화면 너비가 256일 때 중앙

        if center_x > screen_center_x:
            # 객체가 중앙보다 오른쪽에 있으면 왼쪽으로 회피
            print('오른쪽 객체 발견! 왼쪽으로 회피합니다.')
            car_controls.throttle = 0.5
            car_controls.steering = -0.5  # -1.0 대신 작은 값으로 부드럽게


        elif center_x < screen_center_x:
            # 객체가 중앙보다 왼쪽에 있으면 오른쪽으로 회피
            print('왼쪽 객체 발견! 오른쪽으로 회피합니다.')
            car_controls.throttle = 0.5
            car_controls.steering = 0.5  # 1.0 대신 작은 값으로 부드럽게


        else:
            # 객체가 중앙에 있으면 정지
            print('정면에 객체 발견! 정지합니다.')
            car_controls.throttle = 0.0
            car_controls.steering = 0.0
            car_controls.brake = 1.0


    else:
        # 바운딩 박스 크기가 작으면(장애물이 멀리 있으면) 직진
        print('이상없음. 직진합니다.')
        car_controls.throttle = 0.5
        car_controls.steering = 0.0
        car_controls.brake = 0.0
        client.setCarControls(car_controls)


def main():
    try:
        while True:
            # All API calls should be inside the loop and after enabling API control
            client.setCarControls(car_controls)

            if keyboard.is_pressed('up'):  # 위쪽 화살표 키를 누르면
                car_controls.throttle = 1.0
            if keyboard.is_pressed('down'):  # 아래쪽 화살표 키를 누르면
                car_controls.brake = 1.0
            if keyboard.is_pressed('left'):  # 왼쪽 화살표 키를 누르면
                car_controls.steering = -1.0
            if keyboard.is_pressed('right'):  # 오른쪽 화살표 키를 누르면
                car_controls.steering = 1.0

            airsim.time.sleep(0.01)

            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            response = responses[0]
            width = response.width
            height = response.height
            screen_size = width * height
            print("screen_size: {} ".format(screen_size))
            print("width: {}, height: {}".format(width, height))

            # YOLO Detection screen info
            '''
            width = 256, height: 144
            '''
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            results = model(img_bgr, conf=0.5)

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
                    # print("bounding box size: {}".format(bounding_box_size))

                    avoid(bounding_box_size, x1, y1, x2, y2)

                    cv2.putText(img_bgr, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

            cv2.imshow("YOLO Detection", img_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # drive(throttle=0.5, steering=0.0, duration=1)

    finally:
        # Disable API control only once at the end of the program
        client.enableApiControl(False, vehicle_name="Car1")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO('yolov8m.pt')
    client = airsim.CarClient()
    client.confirmConnection()
    car_controls = airsim.CarControls()
    main()
