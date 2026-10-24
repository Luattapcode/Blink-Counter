# 👁️ Real-Time Blink Counter with MediaPipe Tasks API & OpenCV

Ứng dụng phát hiện và đếm số lần chớp mắt thời gian thực (Real-Time Blink Counter) dựa trên xử lý luồng video camera, sử dụng mô hình **MediaPipe Tasks API (FaceLandmarker)** và thuật toán chỉ số **EAR (Eye Aspect Ratio)**.

---

## 📌 Tính năng chính
* 🎥 **Bắt luồng video từ Webcam thời gian thực** với tốc độ FPS cao.
* 📍 **Xác định chính xác điểm mốc (landmarks)** trên khuôn mặt bằng MediaPipe Tasks API (`FaceLandmarker`).
* 📐 **Tính toán chỉ số EAR (Eye Aspect Ratio)** để định lượng chính xác trạng thái đóng/mở của mắt.
* 🔢 **Tự động lọc nhiễu** (chống đếm nhầm do nháy mắt giả) và cập nhật số lần chớp mắt lên màn hình.
* 📦 **Tự động tải file mô hình (`face_landmarker.task`)** nếu chưa có trong thư mục cục bộ.

---

## 🛠️ Yêu cầu hệ thống
* **Hệ điều hành:** Windows / macOS / Linux
* **Python:** Phiên bản `3.9` đến `3.14`
* **Thư viện chính:** OpenCV, MediaPipe, NumPy

---

## 📁 Cấu trúc thư mục dự án

```text
Blink-Counter/
├── .gitignore             # Cấu hình bỏ qua các file rác (__pycache__)
├── requirements.txt       # Danh sách các thư viện phụ thuộc
├── face_landmarker.task   # File trọng số mô hình MediaPipe Tasks API
├── utils.py               # Module chứa hàm tính toán khoảng cách và EAR
├── main.py                # Chương trình chính xử lý camera và logic đếm
└── README.md              # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn cài đặt và chạy chương trình

### Bước 1: Tải mã nguồn về máy (Clone Repository)
Mở Terminal / Command Prompt và chạy lệnh:
```bash
git clone https://github.com/Luattapcode/Blink-Counter.git
cd Blink-Counter
```

---

### Bước 2: Cài đặt trực tiếp các thư viện cần thiết

Mở Terminal / Command Prompt ngay tại thư mục dự án và cài đặt toàn bộ thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu dùng các bản Python mới và gặp sự cố về pip, bạn có thể cập nhật pip trước bằng lệnh: `python -m pip install --upgrade pip`)*

---

### Bước 3: Chạy chương trình

Chạy file chương trình chính:
```bash
python main.py
```

* **Hướng dẫn tương tác:**
  * Đưa mặt vào góc quay của Webcam.
  * Các điểm mốc màu vàng sẽ xuất hiện quanh vùng mắt cùng chỉ số `EAR` và số lần `Blinks` tăng lên theo từng cú chớp mắt.
  * Nhấn phím **`q`** trên bàn phím để thoát chương trình.

---

## 📊 Thuật toán cốt lõi (EAR - Eye Aspect Ratio)

Chỉ số EAR được tính dựa trên tỉ lệ khoảng cách giữa các điểm mốc chiều dọc và chiều ngang của mắt:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$

* **Khi mắt mở:** $\text{EAR} \ge 0.22$.
* **Khi chớp mắt:** $\text{EAR} < 0.22$ duy trì liên tiếp $\ge 2$ khung hình (`CONSEC_FRAMES = 2`) $\rightarrow$ Đếm $+1$ lần chớp mắt.

---

## 💻 Mã nguồn chi tiết dự án (Source Code)

### 1. File `requirements.txt`
```text
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

### 2. File `.gitignore`
__pycache__/


---

### 3. File `utils.py`
```python
import numpy as np

def calc(pt1, pt2):
    """Tính khoảng cách Euclidean giữa 2 điểm 2D."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def eye_aspect_ratio(eye_landmarks):
    """
    Tính chỉ số EAR (Eye Aspect Ratio) dựa trên 6 điểm mốc của mắt.
    """
    v1 = calc(eye_landmarks[1], eye_landmarks[5])
    v2 = calc(eye_landmarks[2], eye_landmarks[4])
    h = calc(eye_landmarks[0], eye_landmarks[3])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear
```

---

### 4. File `main.py`
```python
import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import eye_aspect_ratio

# Tự động tải file mô hình Face Landmarker nếu chưa có trong thư mục
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Đang tải file mô hình MediaPipe Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Khởi tạo FaceLandmarker từ MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Chỉ số mốc mắt chuẩn trong MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2

counter = 0
frame_count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            left_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye_pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_count += 1
            else:
                if frame_count >= CONSEC_FRAMES:
                    counter += 1
                frame_count = 0

            # Hiển thị thông tin lên giao diện video
            cv2.putText(frame, f"Blinks: {counter}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            for pt in left_eye_pts + right_eye_pts:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

    cv2.imshow("Blink Counter Application", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📝 Tác giả
* **Họ và tên:** Huỳnh Quốc Luật
* **GitHub:** [Luattapcode](https://github.com/Luattapcode)
* **Đề tài:** Bài tập lớn môn Nhập Môn Lập Trình
