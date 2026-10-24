import numpy as np

def calc(p1,p2) :
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_landmarks):
    vertical_1 = calc(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = calc(eye_landmarks[2], eye_landmarks[4])
    horizontal = calc(eye_landmarks[0], eye_landmarks[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear