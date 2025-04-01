import cv2
import mediapipe as mp
import numpy as np
import dearpygui.dearpygui as dpg
from enum import Enum, auto

class SignLanguage(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    UNKNOWN = auto()

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
        
        # Display settings
        self.cam_width, self.cam_height = 640, 480
        
        # Initialize GUI
        self.setup_gui()
        self.current_sign = SignLanguage.UNKNOWN
        self.running = True

    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Sign Language Detector', width=800, height=600)
        
        with dpg.texture_registry(show=False):
            self.texture_data = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.float32)
            self.raw_texture = dpg.add_raw_texture(
                width=self.cam_width, height=self.cam_height, 
                default_value=self.texture_data, format=dpg.mvFormat_Float_rgb,
                tag="camera_texture"
            )
        
        with dpg.window(label="Sign Language Detection", tag="main_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=500):
                    dpg.add_image("camera_texture")
                
                with dpg.child_window(width=300):
                    dpg.add_text("Sign Language Detection", color=(0, 255, 255))
                    self.sign_text = dpg.add_text("Current Sign: UNKNOWN", color=(255, 255, 0))
                    self.confidence_text = dpg.add_text("Confidence: 0%", color=(255, 255, 255))
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def detect_sign_language(self, landmarks):
        # Simplified sign language detection logic
        # This is a placeholder - you'll need to implement proper detection
        
        # Get finger landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        wrist = landmarks.landmark[0]
        
        # Basic detection examples (you should expand this)
        if thumb_tip.y < wrist.y and index_tip.y > wrist.y:
            return SignLanguage.A, 0.9
        elif thumb_tip.x < wrist.x and pinky_tip.x > wrist.x:
            return SignLanguage.B, 0.85
        else:
            return SignLanguage.UNKNOWN, 0.0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Detect sign language
            self.current_sign, confidence = self.detect_sign_language(hand_landmarks)
            
            # Update GUI
            dpg.set_value(self.sign_text, f"Current Sign: {self.current_sign.name}")
            dpg.set_value(self.confidence_text, f"Confidence: {confidence*100:.1f}%")
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Update texture
        np.copyto(self.texture_data, cv2.resize(rgb_frame, (self.cam_width, self.cam_height)).astype(np.float32) / 255.0)
        dpg.set_value("camera_texture", self.texture_data)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            self.process_frame(frame)
            dpg.render_dearpygui_frame()
            
            if dpg.is_key_down(dpg.mvKey_Q):
                self.running = False
                
        cap.release()
        self.hands.close()
        dpg.destroy_context()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()

