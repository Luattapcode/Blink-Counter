import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

class FaceMeshDetector:
    def __init__(self, model_path='face_landmarker.task', max_faces=1):
        # Tự động tải file mô hình nhận diện khuôn mặt nếu chưa tồn tại trong thư mục dự án
        if not os.path.exists(model_path):
            print("Đang tải file model Face Landmarker về máy, vui lòng đợi một chút...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Tải model thành công!")
            
        # Khởi tạo FaceLandmarker theo chuẩn API mới
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def find_face_landmarks(self, frame):
        """Xử lý ảnh bằng API mới và trả về danh sách các điểm landmarks của khuôn mặt"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi sang định dạng mp.Image của MediaPipe Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        
        if results.face_landmarks:
            # Trả về tập hợp điểm mốc của khuôn mặt đầu tiên tìm thấy
            return results.face_landmarks[0]
        return None