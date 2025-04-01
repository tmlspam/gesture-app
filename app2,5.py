import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
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
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # Display settings
        self.cam_width, self.cam_height = 640, 480
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Control settings
        self.sensitivity = 0.7
        self.smoothing = 0.3
        self.mouse_positions = np.zeros((5, 2), dtype=np.float32)
        self.pos_idx = 0
        
        # State tracking
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.control_mode = "mouse"
        self.running = True
        
        # Initialize GUI
        self.setup_gui()

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

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        curr_time = time.perf_counter()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get essential landmarks
            landmarks = {
                'wrist': (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y),
                'thumb': (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y),
                'index': (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y),
                'middle': (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y),
                'ring': (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y),
                'pinky': (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y)
            }
            
            # Detect gesture
            self.current_gesture = self._detect_gesture(landmarks)
            self._execute_control(landmarks)
            
            dpg.set_value(self.gesture_text, f"Gesture: {self.current_gesture.name}")
            mp.solutions.drawing_utils.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Update texture
        np.copyto(self.texture_data, cv2.resize(rgb_frame, (self.cam_width, self.cam_height)).astype(np.float32) / 255.0)
        dpg.set_value("camera_texture", self.texture_data)
        
        return curr_time

    def _detect_gesture(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return Gesture.NONE
            
        # Movement detection
        curr_pos = np.array([landmarks['wrist'][0], landmarks['wrist'][1]])
        prev_pos = np.array([self.prev_landmarks['wrist'][0], self.prev_landmarks['wrist'][1]])
        movement = curr_pos - prev_pos
        velocity = np.linalg.norm(movement)
        
        # Swipe detection
        if velocity > 0.1 * (2 - self.sensitivity):
            if abs(movement[0]) > abs(movement[1]):
                return Gesture.SWIPE_RIGHT if movement[0] > 0 else Gesture.SWIPE_LEFT
            else:
                return Gesture.SWIPE_DOWN if movement[1] > 0 else Gesture.SWIPE_UP
        
        # Thumb up detection (for right click)
        thumb_up = (landmarks['thumb'][1] < landmarks['wrist'][1] - 0.1 and 
                   landmarks['index'][1] > landmarks['wrist'][1] and
                   landmarks['middle'][1] > landmarks['wrist'][1])
        
        if thumb_up:
            return Gesture.THUMB_UP
        
        # Pinch detection
        thumb_index_dist = np.linalg.norm(np.array(landmarks['thumb']) - np.array(landmarks['index']))
        if thumb_index_dist < 0.05 * (2 - self.sensitivity):
            return Gesture.PINCH
            
        return Gesture.POINTING if landmarks['index'][1] < landmarks['wrist'][1] else Gesture.NONE

    def _execute_control(self, landmarks):
        if self.control_mode == "mouse":
            self._control_mouse(landmarks)
        elif self.control_mode == "keyboard":
            self._control_keyboard()

    def _control_mouse(self, landmarks):
        # Calculate screen position
        screen_pos = (
            int((landmarks['index'][0] - 0.5) * self.screen_width * self.sensitivity * 2 + self.screen_width/2),
            int((landmarks['index'][1] - 0.5) * self.screen_height * self.sensitivity * 2 + self.screen_height/2)
        )
        
        # Smooth movement
        self.mouse_positions[self.pos_idx] = screen_pos
        self.pos_idx = (self.pos_idx + 1) % len(self.mouse_positions)
        avg_pos = np.mean(self.mouse_positions, axis=0).astype(int)
        
        pyautogui.moveTo(*avg_pos, _pause=False)
        
        # Gesture actions
        if self.current_gesture == Gesture.PINCH:
            pyautogui.click(_pause=False)
        elif self.current_gesture == Gesture.THUMB_UP:
            pyautogui.rightClick(_pause=False)

    def _control_keyboard(self):
        if self.current_gesture == Gesture.SWIPE_UP:
            pyautogui.press('volumeup', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_DOWN:
            pyautogui.press('volumedown', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_LEFT:
            pyautogui.press('left', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_RIGHT:
            pyautogui.press('right', _pause=False)

    def run(self):
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

if __name__ == "__main__":
    controller = GestureController()
    controller.run()