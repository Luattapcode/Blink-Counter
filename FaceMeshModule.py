import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, max_faces=1, refine_landmarks=True):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Khởi tạo model FaceMesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def find_face_landmarks(self, frame):
        """Đưa ảnh BGR vào, trả về bộ landmarks của khuôn mặt đầu tiên tìm thấy"""
        # Chuyển đổi hệ màu từ BGR sang RGB cho MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None