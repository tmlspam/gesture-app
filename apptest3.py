import cv2
import mediapipe as mp
import numpy as np
import dearpygui.dearpygui as dpg
import time
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
    Z = auto()
    SPACE = auto()
    UNKNOWN = auto()

class AdvancedASLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.cam_width, self.cam_height = 640, 480
        self.current_sign = SignLanguage.UNKNOWN
        self.sign_history = []
        self.running = True
        self.last_time = time.time()
        
        self.setup_gui()
        
    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Advanced ASL Detector', width=1000, height=800)
        
        with dpg.texture_registry(show=False):
            self.texture_data = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.float32)
            dpg.add_raw_texture(
                width=self.cam_width, height=self.cam_height,
                default_value=self.texture_data, format=dpg.mvFormat_Float_rgb,
                tag="camera_texture"
            )
        
        with dpg.window(label="ASL Detection", tag="main_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=600):
                    dpg.add_image("camera_texture")
                    self.fps_text = dpg.add_text("FPS: 0")
                    self.detection_text = dpg.add_text("Detected: UNKNOWN", color=(0, 255, 255))
                with dpg.child_window(width=380):
                    self.sign_display = dpg.add_text("Sign: ", tag="sign_display")
                    self.history_display = dpg.add_text("History:", tag="history_display")
                    dpg.add_button(label="Clear History", callback=self.clear_history)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def detect_asl_sign(self, hand_landmarks):
        joints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        fingers = [
            joints[8][1] < joints[6][1],  # Index
            joints[12][1] < joints[10][1], # Middle
            joints[16][1] < joints[14][1], # Ring
            joints[20][1] < joints[18][1]  # Pinky
        ]
        if all(fingers):
            return SignLanguage.B, 0.95
        elif fingers[0] and not any(fingers[1:]):
            return SignLanguage.D, 0.85
        else:
            return SignLanguage.UNKNOWN, 0.0
    
    def clear_history(self):
        self.sign_history = []
        dpg.set_value("history_display", "History:")
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_sign, confidence = self.detect_asl_sign(hand_landmarks)
            dpg.set_value("sign_display", f"Sign: {self.current_sign.name}")
            self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
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
    detector = AdvancedASLDetector()
    detector.run()


