클라이언트  생성
client = airsim.CarClient()
client.confirmConnection()  # 연결 확인
client.enableApiControl(True)  # Python API 제어 활성화
car_controls = airsim.CarControls()

차량 위지 정보
car_state = client.getCarState()
print(car_state.kinematics_estimated.position)

정면 LIDAR 값 가져오기
lidar_data = client.getLidarData(lidar_name="LidarSensor1")
points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

print("LiDAR 포인트 수:", points.shape[0])