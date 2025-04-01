import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import threading
from enum import Enum, auto
import dearpygui.dearpygui as dpg

class Gesture(Enum):
    NONE = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    PINCH = auto()
    POINTING = auto()
    THUMB_UP = auto()

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        self.cam_width, self.cam_height = 640, 480
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.sensitivity = 0.7
        self.smoothing = 0.3
        self.mouse_positions = np.zeros((5, 2), dtype=np.float32)
        self.pos_idx = 0
        
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.control_mode = "mouse"
        self.running = True
        
        self.setup_gui()
        self.capture_thread = threading.Thread(target=self.run_capture, daemon=True)
        self.capture_thread.start()
    
    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title='Gesture Control Pro', width=800, height=600)
        
        with dpg.texture_registry(show=False):
            self.texture_data = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.float32)
            self.raw_texture = dpg.add_raw_texture(
                width=self.cam_width, height=self.cam_height, 
                default_value=self.texture_data, format=dpg.mvFormat_Float_rgb,
                tag="camera_texture"
            )
        
        with dpg.window(label="Control Panel", tag="main_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=400):
                    dpg.add_image("camera_texture")
                
                with dpg.child_window(width=400):
                    dpg.add_text("Gesture Control Pro", color=(0, 255, 255))
                    
                    with dpg.collapsing_header(label="Settings", default_open=True):
                        dpg.add_slider_float(
                            label="Sensitivity", min_value=0.1, max_value=1.0,
                            default_value=self.sensitivity, callback=lambda s, d: setattr(self, 'sensitivity', d)
                        )
                        dpg.add_slider_float(
                            label="Smoothing", min_value=0.0, max_value=1.0,
                            default_value=self.smoothing, callback=lambda s, d: setattr(self, 'smoothing', d)
                        )
                    
                    with dpg.group(horizontal=True):
                        self.status_text = dpg.add_text("Mouse Mode", color=(0, 255, 0))
                        self.gesture_text = dpg.add_text("Gesture: None", color=(255, 255, 0))
                    
                    self.fps_text = dpg.add_text("FPS: 0")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Mouse", callback=lambda: self.set_control_mode("mouse"), width=100)
                        dpg.add_button(label="Media", callback=lambda: self.set_control_mode("media"), width=100)
                        dpg.add_button(label="Keyboard", callback=lambda: self.set_control_mode("keyboard"), width=100)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def set_control_mode(self, mode):
        self.control_mode = mode
        dpg.set_value(self.status_text, f"{mode.capitalize()} Mode")
    
    def run_capture(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prev_time = time.perf_counter()
        frame_count = 0
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            curr_time = self.process_frame(frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                fps = 10 / (curr_time - prev_time)
                dpg.set_value(self.fps_text, f"FPS: {int(fps)}")
                prev_time = curr_time
            
            dpg.render_dearpygui_frame()
            if dpg.is_key_down(dpg.mvKey_Q):
                self.running = False
        
        cap.release()
        self.hands.close()
        dpg.destroy_context()
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        curr_time = time.perf_counter()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            landmarks = {f"p{i}": (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}
            self.current_gesture = self._detect_gesture(landmarks)
            self._execute_control(landmarks)
            
            dpg.set_value(self.gesture_text, f"Gesture: {self.current_gesture.name}")
            mp.solutions.drawing_utils.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        np.copyto(self.texture_data, cv2.resize(rgb_frame, (self.cam_width, self.cam_height)).astype(np.float32) / 255.0)
        dpg.set_value("camera_texture", self.texture_data)
        
        return curr_time
    
    def _detect_gesture(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return Gesture.NONE
        
        return Gesture.POINTING
    
    def _execute_control(self, landmarks):
        if self.control_mode == "mouse":
            pyautogui.moveTo(int(landmarks['p8'][0] * self.screen_width), int(landmarks['p8'][1] * self.screen_height), _pause=False)
    
if __name__ == "__main__":
    controller = GestureController()
