import math

# Chỉ số các điểm mốc của mắt theo chuẩn MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm (x, y)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_ear(eye_points, landmarks, img_w, img_h):
    """Tính toán chỉ số Tỷ lệ khung mắt (EAR) cho API mới"""
    # Ở API mới, landmarks là list trực tiếp, gọi landmarks[p].x thay vì .landmark[p].x
    pts = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in eye_points]
    
    # Tính chiều dọc (v1, v2) và chiều ngang (h_dist)
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    h_dist = euclidean_distance(pts[0], pts[3])
    
    if h_dist == 0:
        return 0.0
    
    # Công thức tính EAR
    return (v1 + v2) / (2.0 * h_dist)