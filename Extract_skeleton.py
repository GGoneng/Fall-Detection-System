# -----------------------------------------------------------------------------------
# 파일명       : Extract_skeleton.py
# 설명         : 영상 데이터에서 관절 스켈레톤 데이터 추출
# 작성자       : 이민하
# 작성일       : 2025-05-02
# 
# 사용 모듈    :
# - os           # 파일 및 경로 관리
# - cv2          # OpenCV를 활용한 이미지 및 비디오 처리
# - mediapipe    # MediaPipe Pose 모델을 사용한 관절 데이터 추출
# - csv          # 데이터를 CSV 형식으로 저장
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 지정한 관절 좌표를 영상 데이터에서 추출하여 CSV 파일로 저장
# -----------------------------------------------------------------------------------

# 파일 및 경로 관리
import os

# OpenCV를 활용한 이미지 및 비디오 처리
import cv2

# MediaPipe Pose 모델을 사용한 관절 데이터 추출
import mediapipe as mp

# 데이터를 CSV 형식으로 저장
import csv

# 영상 데이터 경로 지정 함수
def extract_path(PATH):
    # 경로를 담을 리스트
    path_list = []

    # 데이터 경로의 MP4파일 추출해서 리스트 저장
    for dirpath, _, filenames in os.walk(PATH):
        for filename in filenames:
            if filename.endswith((".mp4", ".MP4")):
                path_list.append(os.path.join(dirpath, filename))

    # 갯수 제한 (1500개)
    return path_list

# 스켈레톤 데이터 추출
def extract_csv(path_list):
    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        # 학습을 위해 최대한 정확도를 높게 추출
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True
    )

    # 중요 관절 인덱스
    # 0 - 코, 11 - 왼쪽 어깨, 12 - 오른쪽 어깨, 13 - 왼족 팔꿈치, 14 - 오른쪽 팔꿈치
    # 15 - 왼쪽 허리, 16 - 오른쪽 허리, 23 - 왼쪽 엉덩이, 24 - 오른쪽 엉덩이
    # 25 - 왼쪽 무릎, 26 - 오른쪽 무릎, 29 - 왼쪽 뒷꿈치, 30 - 오른쪽 뒷꿈치
    important_landmarks = [
        0, 11, 12, 13, 14, 15, 16,
        23, 24, 25, 26, 29, 30
    ]

    # 영상 데이터별로 좌표 추출 시작
    for i, path in enumerate(path_list):
        cap = cv2.VideoCapture(path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # 20fps 추출로 제한
        skip_interval = max(1, round(original_fps / 20))


        # 0 ~ 1500 : 낙상
        # 1501 ~  : 비낙상
        # 추출할 때마다 CSV 파일로 저장
        with open(f'./Dataset/pose_landmark_{2069 + i:04d}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # 각 랜드마크(관절)별로 x, y, z 좌표 추출
            writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z'])

            frame_count = 0
            processed_frame = 1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % skip_interval == 0:
                    # 프레임 크기가 작을수록 정확도 감소로 인해 큰 사이즈 유지
                    frame = cv2.resize(frame, (1920, 1280))
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 좌표 추출
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        for idx in important_landmarks:
                            landmark = results.pose_landmarks.landmark[idx]
                            # 좌표값 CSV파일에 작성
                            writer.writerow([processed_frame, idx, landmark.x, landmark.y, landmark.z])

                        # 좌표값 인식 디버깅
                        # 실행 시, 추출 속도 저하로 인해 주석 처리
                        # for idx in important_landmarks:
                        #     landmark = results.pose_landmarks.landmark[idx]
                        #     x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        #     cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
                    processed_frame += 3

                    # cv2.imshow('Pose with Important Landmarks', frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print("CSV 파일과 비디오 파일로 저장 완료.")


if __name__ == "__main__":
    # 영상 데이터 위치 설정
    PATH = r"F:/Fall_Detection_Data/Source_Data/Video/N2"
    # PATH = r"./Test_Dataset/"
    # PATH = r"F:/Fall_Detection_Data/Source_Data/Video/"
    
    # 영상 데이터 경로 추출
    path_list = extract_path(PATH)

    # print(path_list)
    
    # 스켈레톤 데이터 추출
    extract_csv(path_list)