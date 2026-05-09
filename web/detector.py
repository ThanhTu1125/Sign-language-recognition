import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode, 
            max_num_hands=max_hands,
            min_detection_confidence=detection_con, 
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp_drawing

    def find_hand_landmarks(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_landmarks = []

        h, w, c = img.shape 

        multi_landmarks = getattr(self.results, 'multi_hand_landmarks', None)

        if multi_landmarks:
            for hand_lms in multi_landmarks:
                if draw and self.mp_draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_lms, 
                        list(self.mp_hands.HAND_CONNECTIONS)
                    )
                
                for lm in hand_lms.landmark:
                    all_landmarks.extend([lm.x * w, lm.y * h, lm.z * w])
        
        return img, all_landmarks