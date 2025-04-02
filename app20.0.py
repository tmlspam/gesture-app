import cv2
import mediapipe as mp
import pyttsx3
import numpy as np

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize Text-to-Speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Slower speech speed
        
        # Finger landmarks
        self.tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.mcp = [2, 5, 9, 13, 17]    # Finger base joints
        
        # To avoid repeating the same letter
        self.last_spoken_letter = None
    
    def get_finger_state(self, hand_landmarks):
        """Returns list of extended finger states (True = extended)"""
        return [
            hand_landmarks.landmark[self.tips[i]].y < 
            hand_landmarks.landmark[self.mcp[i]].y
            for i in range(5)
        ]
    
    def detect_letter(self, extended):
        """Enhanced letter detection with more signs"""
        thumb, index, middle, ring, pinky = extended
        
        if not any(extended[1:]):  # Only thumb might be extended
            return "A" if thumb else "Closed Fist"
        elif all(extended[1:]) and not thumb:
            return "B"
        elif index and middle and not (ring or pinky):
            return "C"
        elif index and not any([middle, ring, pinky]):
            return "D"
        elif all(extended):
            return "E"
        elif thumb and pinky and not any([index, middle, ring]):
            return "Y"
        elif index and thumb and not any([middle, ring, pinky]):
            return "L"
        else:
            return None
    
    def speak_letter(self, letter):
        """Speak the detected letter if it's new"""
        if letter and letter != self.last_spoken_letter:
            self.tts_engine.say(letter)
            self.tts_engine.runAndWait()
            self.last_spoken_letter = letter
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                extended = self.get_finger_state(hand_landmarks)
                letter = self.detect_letter(extended)
                
                if letter:
                    cv2.putText(frame, f"Letter: {letter}", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.speak_letter(letter)
        return frame

def main():
    detector = SignLanguageDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = detector.process_frame(frame)
        cv2.imshow('Sign Language Detector', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
