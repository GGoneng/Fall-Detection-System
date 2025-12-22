import torch
import mediapipe as mp
import time
import torch
import mediapipe as mp
import time
import pandas as pd
import numpy as np
from collections import deque
from picamera2 import Picamera2
import cv2
import datetime
import uuid
import hmac
import hashlib
import requests
import platform
import RPi.GPIO as GPIO
import subprocess  # WAV 재생용
import threading

def unique_id():
    return str(uuid.uuid1().hex)

def get_iso_datetime():
    utc_offset_sec = time.altzone if time.localtime().tm_isdst else time.timezone
    utc_offset = datetime.timedelta(seconds=-utc_offset_sec)
    return datetime.datetime.now().replace(tzinfo=datetime.timezone(offset=utc_offset)).isoformat()

def get_signature(key, msg):
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()

def get_headers(api_key, api_secret):
    date = get_iso_datetime()
    salt = unique_id()
    combined_string = date + salt
    return {
        'Authorization': 'HMAC-SHA256 ApiKey=' + api_key + ', Date=' + date + ', salt=' + salt + ', signature=' +
                         get_signature(api_secret, combined_string),
        'Content-Type': 'application/json; charset=utf-8'
    }

def get_url(path):
    url = '%s://%s' % (protocol, domain)
    if prefix != '':
        url = url + prefix
    url = url + path
    return url

def send_many(parameter):
    api_key = 'NCSWV3C1HU3OZXOG'
    api_secret = 'BKNFXW3RP1DFSTZ3ZWDVFY6ZPF2W71PP'
    parameter['agent'] = {
        'sdkVersion': 'python/4.2.0',
        'osPlatform': platform.platform() + " | " + platform.python_version()
    }
    return requests.post(get_url('/messages/v4/send-many'), headers=get_headers(api_key, api_secret), json=parameter)

# 서보 각도 설정 함수
def set_servo_angle(servo, angle):
    duty = 2.5 + (angle / 180) * 10
    servo.ChangeDutyCycle(duty)
    time.sleep(0.05)
    servo.ChangeDutyCycle(0)

def pose_tracking():
    global left_shoulder, right_shoulder, left_wrist, right_wrist, nose, left_ankle, right_ankle

    # PiCamera2 설정
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (480, 360)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    # 초기 Pan, Tilt 각도 설정
    pan_angle = 90
    tilt_angle = 90
    set_servo_angle(pan_servo, pan_angle)
    set_servo_angle(tilt_servo, tilt_angle)

    start_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed >= 0.1:
                start_time = current_time
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_wrist = landmarks[15]
                    right_wrist = landmarks[16]
                    nose = landmarks[0]
                    left_ankle = landmarks[27]
                    right_ankle = landmarks[28]
                                    

                    upper_body_detected = (left_shoulder.visibility > 0.5 or right_shoulder.visibility > 0.5)

                    if upper_body_detected:
                        upper_body_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]
                        upper_body_center_y = (left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]
                        center_x = frame.shape[1] / 2
                        center_y = frame.shape[0] / 2
                        x_threshold = 20
                        y_threshold = 20

                        if upper_body_center_x < center_x - x_threshold:
                            pan_angle = min(180, pan_angle + 2)
                            print("상체 왼쪽: 카메라 오른쪽으로 이동")
                        elif upper_body_center_x > center_x + x_threshold:
                            pan_angle = max(0, pan_angle - 2)
                            print("상체 오른쪽: 카메라 왼쪽으로 이동")
                        else:
                            print("Pan: 중앙 유지")

                        if upper_body_center_y < center_y - y_threshold:
                            tilt_angle = max(0, tilt_angle - 2)
                            print("상체 위쪽: 카메라 위로 이동")
                        elif upper_body_center_y > center_y + y_threshold:
                            tilt_angle = min(180, tilt_angle + 2)
                            print("상체 아래쪽: 카메라 아래로 이동")
                        else:
                            print("Tilt: 중앙 유지")

                    left_arm_detected = left_wrist.visibility > 0.5 and right_wrist.visibility <= 0.5
                    right_arm_detected = right_wrist.visibility > 0.5 and left_wrist.visibility <= 0.5
                    head_detected = nose.visibility > 0.5 and not upper_body_detected
                    leg_detected = (left_ankle.visibility > 0.5 or right_ankle.visibility > 0.5) and not upper_body_detected

                    if left_arm_detected and not upper_body_detected:
                        pan_angle = min(180, pan_angle + 3)
                        print("왼팔만 인식: 상체 인식 전까지 오른쪽 이동")
                    elif right_arm_detected and not upper_body_detected:
                        pan_angle = max(0, pan_angle - 3)
                        print("오른팔만 인식: 상체 인식 전까지 왼쪽 이동")

                    if head_detected and not upper_body_detected:
                        tilt_angle = min(180, tilt_angle + 2)
                        print("머리만 인식: 상체 인식 전까지 아래로 이동")
                    elif leg_detected and not upper_body_detected:
                        tilt_angle = max(0, tilt_angle - 2)
                        print("다리만 인식: 상체 인식 전까지 위로 이동")

                    duty_pan = 2.5 + (pan_angle / 180) * 10
                    duty_tilt = 2.5 + (tilt_angle / 180) * 10
                    pan_servo.ChangeDutyCycle(duty_pan)
                    tilt_servo.ChangeDutyCycle(duty_tilt)
                    time.sleep(0.05)
                    pan_servo.ChangeDutyCycle(0)
                    tilt_servo.ChangeDutyCycle(0)

                    for idx in important_landmarks:
                        landmark = results.pose_landmarks.landmark[idx]
                        window.append((landmark.x, landmark.y, landmark.z))
                else:
                    for idx in important_landmarks:
                        window.append((0, 0, 0))

                cv2.imshow("Camera View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if output == 1:
                    stop_event.set()
                    break

    except KeyboardInterrupt:
        print("프로그램 중지됨.")

    finally:
        picam2.stop()
        pan_servo.stop()
        tilt_servo.stop()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        pose.close()

def fall_detecting():
    global output

    output = 0

    while True:
        time.sleep(5)

        with lock:
            if len(window) < 13 * 50:
                continue
            
            window_list = list(window)

        index_list = []
        j = 1
        for i in range(len(window_list)):
            index_list.append(j - 1)
            j += 1
            if j % 13 == 1:
                j += 26

        test_df.loc[index_list, ['x', 'y', 'z']] = window_list

        test_df2 = test_df.groupby("landmark_id").apply(
            lambda x: x.sort_values(by=["frame", "landmark_id"]).interpolate().ffill()
        ).reset_index(drop=True)
        test_df2 = test_df2.sort_values(by=["frame", "landmark_id"]).reset_index(drop=True)

        df_pivot = test_df2.pivot(index="frame", columns="landmark_id", values=["x", "y", "z"])
        df_pivot.columns = [f"{coord}_{int(lid)}" for coord, lid in df_pivot.columns]
        df_pivot.reset_index(inplace=True)
        df_pivot = df_pivot.sort_values(by=["frame"]).reset_index(drop=True)

        X_tensor = torch.tensor(df_pivot.values, dtype=torch.float32).unsqueeze(0)
        model_output = torch.sigmoid(fall_detect_model(X_tensor))
        output = (model_output > 0.5).int()

        if output == 1:
            data = {
                'messages': [
                    {
                        'to': '',
                        'from': '',
                        'text': 'fall detected!!!'
                    }
                ]
            }
                
            response = send_many(data)
            print("낙상 감지됨!")
            stop_event.set()
            break

        else:
            print("낙상 없음:", output.item())

def alarm():    
    while not stop_event.is_set():
        if output == 1:
            subprocess.run(["aplay", "-D", "plughw:3,0", "fall_detected.wav"])

            for _ in range(5):
                buzzer_pwm.start(50)
                time.sleep(0.2)
                buzzer_pwm.stop()
                time.sleep(0.2)

            stop_event.set()
            break
        time.sleep(0.1)

if __name__ == "__main__":
    # SMS 전송 관련 설정
    protocol = 'https'
    domain = 'api.coolsms.co.kr'
    prefix = ''

    # 학습된 낙상 감지 모델 불러오기
    fall_detect_model = torch.jit.load("model_script (128).pt")

    lock = threading.Lock()
    stop_event = threading.Event()

    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=False
    )
    mp_drawing = mp.solutions.drawing_utils

    # 중요 랜드마크 정의
    important_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30]
    MAX_FRAMES_LEN = 300
    window = deque(maxlen=((MAX_FRAMES_LEN // 3) * len(important_landmarks)))

    # 데이터 프레임 템플릿 생성
    df_data = np.empty((MAX_FRAMES_LEN * len(important_landmarks), 5), dtype=np.float32)
    df_data[:] = np.nan
    row_index = 0
    for frame in range(1, MAX_FRAMES_LEN + 1, 3):
        for landmark in important_landmarks:
            df_data[row_index] = [frame, landmark, np.nan, np.nan, np.nan]
            row_index += 1
    test_df = pd.DataFrame(df_data, columns=["frame", "landmark_id", "x", "y", "z"])


    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(12, GPIO.OUT)  # 부저
    GPIO.setup(13, GPIO.OUT)  # Pan 서보
    GPIO.setup(23, GPIO.OUT)  # Tilt 서보

    # PWM 설정
    buzzer_pwm = GPIO.PWM(12, 2000)
    pan_servo = GPIO.PWM(13, 50)
    tilt_servo = GPIO.PWM(23, 50)

    pan_servo.start(0)
    tilt_servo.start(0)

    buzzer_pwm.stop()

    pose_thread = threading.Thread(target = pose_tracking)
    detect_thread = threading.Thread(target = fall_detecting)
    alarm_thread = threading.Thread(target = alarm)

    pose_thread.start()
    detect_thread.start()
    alarm_thread.start()

    pose_thread.join()
    detect_thread.join()
    alarm_thread.join()
    