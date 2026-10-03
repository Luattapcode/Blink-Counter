import cv2
# Import các module đã viết từ các file khác
from FaceMeshModule import FaceMeshDetector
from utils import get_ear, LEFT_EYE, RIGHT_EYE

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    
    # Biến trạng thái
    EAR_THRESHOLD = 0.22  # Ngưỡng nhắm mắt (điều chỉnh tùy mắt mỗi người)
    blink_count = 0
    is_closed = False     # Trạng thái hiện tại

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape
        
        # 1. Tìm khuôn mặt
        landmarks = detector.find_face_landmarks(frame)
        
        if landmarks:
            # 2. Tính EAR cho cả 2 mắt
            left_ear = get_ear(LEFT_EYE, landmarks, img_w, img_h)
            right_ear = get_ear(RIGHT_EYE, landmarks, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # 3. Logic Máy trạng thái (State Machine) đếm nháy mắt
            if avg_ear < EAR_THRESHOLD:
                is_closed = True
                cv2.putText(frame, "Mat dang nham!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if is_closed:
                    # Hoàn thành 1 chu trình nhắm -> mở
                    blink_count += 1
                    is_closed = False
            
            # Vẽ thông số lên màn hình
            cv2.putText(frame, f"Blinks: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Blink Counter Project", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()