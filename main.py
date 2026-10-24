import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import eye_aspect_ratio


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Đang tải file mô hình MediaPipe Face Landmarker (vui lòng đợi trong giây lát)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Tải mô hình thành công!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


EAR_THRESHOLD = 0.22  
CONSEC_FRAMES = 2     

counter = 0      
frame_count = 0  

cap = cv2.VideoCapture(0)
print("Đang chạy ứng dụng Blink-Counter... Nhấn phím 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể đọc dữ liệu từ camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            left_eye_pts = []
            for idx in LEFT_EYE:
                landmark = face_landmarks[idx]
                left_eye_pts.append((int(landmark.x * w), int(landmark.y * h)))

            right_eye_pts = []
            for idx in RIGHT_EYE:
                landmark = face_landmarks[idx]
                right_eye_pts.append((int(landmark.x * w), int(landmark.y * h)))

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_count += 1
            else:
                if frame_count >= CONSEC_FRAMES:
                    counter += 1  
                frame_count = 0   

            cv2.putText(frame, f"Blinks: {counter}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            for pt in left_eye_pts:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)
            for pt in right_eye_pts:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

    cv2.imshow("Blink Counter Application", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()