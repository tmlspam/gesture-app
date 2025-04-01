import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import threading
from enum import Enum, auto
import dearpygui.dearpygui as dpg
from collections import deque

class Gesture(Enum):
    NONE = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    PINCH = auto()
    POINTING = auto()
    THUMB_UP = auto()
    THUMB_DOWN = auto()

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.cam_width, self.cam_height = 640, 480
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Configuration parameters
        self.sensitivity = 0.7
        self.smoothing = 0.3
        self.gesture_detection_threshold = 0.1
        self.swipe_velocity_threshold = 0.5
        
        # State tracking
        self.mouse_positions = deque(maxlen=5)
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.last_gesture_time = time.time()
        self.last_action_time = time.time()
        self.control_mode = "mouse"
        self.running = True
        self.calibrating = False
        self.hand_size = 0.1  # Default hand size
        
        # Initialize GUI
        self.setup_gui()
        
        # Start capture thread
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
                # Camera feed
                with dpg.child_window(width=400):
                    dpg.add_image("camera_texture")
                
                # Control panel
                with dpg.child_window(width=400):
                    dpg.add_text("Gesture Control Pro", color=(0, 255, 255))
                    
                    # Settings section
                    with dpg.collapsing_header(label="Settings", default_open=True):
                        dpg.add_slider_float(
                            label="Sensitivity", min_value=0.1, max_value=1.0,
                            default_value=self.sensitivity, callback=lambda s, d: setattr(self, 'sensitivity', d)
                        )
                        dpg.add_slider_float(
                            label="Smoothing", min_value=0.0, max_value=1.0,
                            default_value=self.smoothing, callback=lambda s, d: setattr(self, 'smoothing', d)
                        )
                        dpg.add_slider_float(
                            label="Gesture Threshold", min_value=0.01, max_value=0.2,
                            default_value=self.gesture_detection_threshold,
                            callback=lambda s, d: setattr(self, 'gesture_detection_threshold', d)
                        )
                    
                    # Status indicators
                    with dpg.group(horizontal=True):
                        self.status_text = dpg.add_text("Mouse Mode", color=(0, 255, 0))
                        self.gesture_text = dpg.add_text("Gesture: None", color=(255, 255, 0))
                    
                    self.fps_text = dpg.add_text("FPS: 0")
                    self.calibration_text = dpg.add_text("", color=(255, 165, 0))
                    
                    # Mode selection buttons
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Mouse", callback=lambda: self.set_control_mode("mouse"), width=100)
                        dpg.add_button(label="Media", callback=lambda: self.set_control_mode("media"), width=100)
                        dpg.add_button(label="Keyboard", callback=lambda: self.set_control_mode("keyboard"), width=100)
                    
                    # Calibration button
                    dpg.add_button(label="Calibrate", callback=self.start_calibration, width=100)
                    
                    # Gesture visualization
                    with dpg.collapsing_header(label="Gesture Visualization"):
                        self.gesture_visual = dpg.add_text("No active gesture detected", wrap=380)
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def start_calibration(self):
        self.calibrating = True
        self.calibration_step = 0
        self.calibration_values = []
        dpg.set_value(self.calibration_text, "Calibration: Show your open hand to the camera")
    
    def complete_calibration(self, hand_size):
        self.hand_size = hand_size
        self.calibrating = False
        dpg.set_value(self.calibration_text, f"Calibration complete. Hand size: {hand_size:.2f}")
    
    def set_control_mode(self, mode):
        self.control_mode = mode
        dpg.set_value(self.status_text, f"{mode.capitalize()} Mode")
        self._update_gesture_visualization()
    
    def run_capture(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prev_time = time.perf_counter()
        frame_count = 0
        
        try:
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
                if dpg.is_key_down(dpg.mvKey_Escape):
                    self.running = False
        except Exception as e:
            print(f"Error in capture thread: {e}")
        finally:
            cap.release()
            self.hands.close()
            if dpg.is_dearpygui_running():
                dpg.destroy_context()
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        curr_time = time.perf_counter()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = {f"p{i}": (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}
            
            if self.calibrating:
                self._handle_calibration(landmarks)
            else:
                self.current_gesture = self._detect_gesture(landmarks)
                self._execute_control(landmarks)
                self._update_gesture_visualization()
            
            mp.solutions.drawing_utils.draw_landmarks(rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        np.copyto(self.texture_data, cv2.resize(rgb_frame, (self.cam_width, self.cam_height)).astype(np.float32) / 255.0)
        dpg.set_value("camera_texture", self.texture_data)
        
        return curr_time
    
    def _handle_calibration(self, landmarks):
        if self.calibration_step == 0:
            # Measure open hand size
            wrist = landmarks['p0']
            middle_tip = landmarks['p12']
            hand_size = np.linalg.norm(np.array(middle_tip) - np.array(wrist))
            self.calibration_values.append(hand_size)
            
            if len(self.calibration_values) >= 10:
                avg_size = np.mean(self.calibration_values)
                self.complete_calibration(avg_size)
    
    def _detect_gesture(self, landmarks):
        thumb_tip = landmarks['p4']
        index_tip = landmarks['p8']
        middle_tip = landmarks['p12']
        wrist = landmarks['p0']
        
        # Dynamic threshold based on calibrated hand size
        pinch_threshold = 0.03 + (self.hand_size * 0.08)
        
        # Calculate distances
        thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        
        # Detect pinch
        if thumb_index_dist < pinch_threshold:
            return Gesture.PINCH
        
        # Detect pointing
        index_extended = (index_tip[1] < landmarks['p6'][1]) and \
                        (index_tip[1] < landmarks['p10'][1]) and \
                        (index_tip[1] < landmarks['p14'][1])
        
        if index_extended:
            middle_closed = middle_tip[1] > landmarks['p10'][1]
            ring_closed = landmarks['p16'][1] > landmarks['p13'][1]
            pinky_closed = landmarks['p20'][1] > landmarks['p17'][1]
            
            if middle_closed and ring_closed and pinky_closed:
                return Gesture.POINTING
        
        # Detect swipes
        if self.prev_landmarks:
            dt = time.time() - self.last_gesture_time
            if dt > 0:  # Prevent division by zero
                dx = landmarks['p8'][0] - self.prev_landmarks['p8'][0]
                dy = landmarks['p8'][1] - self.prev_landmarks['p8'][1]
                velocity_x = abs(dx) / dt
                velocity_y = abs(dy) / dt
                
                if velocity_x > self.swipe_velocity_threshold and abs(dx) > self.gesture_detection_threshold:
                    return Gesture.SWIPE_LEFT if dx < 0 else Gesture.SWIPE_RIGHT
                if velocity_y > self.swipe_velocity_threshold and abs(dy) > self.gesture_detection_threshold:
                    return Gesture.SWIPE_UP if dy < 0 else Gesture.SWIPE_DOWN
        
        # Detect thumb gestures
        thumb_up = (thumb_tip[1] < landmarks['p3'][1]) and \
                  (thumb_tip[1] < landmarks['p2'][1])
        thumb_down = (thumb_tip[1] > landmarks['p3'][1]) and \
                    (thumb_tip[1] > landmarks['p2'][1])
        
        if thumb_up:
            return Gesture.THUMB_UP
        elif thumb_down:
            return Gesture.THUMB_DOWN
        
        self.prev_landmarks = landmarks
        self.last_gesture_time = time.time()
        return Gesture.NONE
    
    def _execute_control(self, landmarks):
        current_time = time.time()
        action_delay = 0.3  # Minimum time between actions
        
        if current_time - self.last_action_time < action_delay:
            return
        
        try:
            if self.control_mode == "mouse":
                if self.current_gesture == Gesture.POINTING:
                    new_pos = [
                        landmarks['p8'][0] * self.screen_width * self.sensitivity,
                        landmarks['p8'][1] * self.screen_height * self.sensitivity
                    ]
                    
                    if hasattr(self, 'last_mouse_pos'):
                        smoothed_pos = [
                            self.smoothing * new_pos[0] + (1 - self.smoothing) * self.last_mouse_pos[0],
                            self.smoothing * new_pos[1] + (1 - self.smoothing) * self.last_mouse_pos[1]
                        ]
                    else:
                        smoothed_pos = new_pos
                    
                    pyautogui.moveTo(int(smoothed_pos[0]), int(smoothed_pos[1]), _pause=False)
                    self.last_mouse_pos = smoothed_pos
                
                elif self.current_gesture == Gesture.PINCH:
                    pyautogui.click()
                    self.last_action_time = current_time
                
                elif self.current_gesture == Gesture.THUMB_UP:
                    pyautogui.rightClick()
                    self.last_action_time = current_time
            
            elif self.control_mode == "media":
                if self.current_gesture == Gesture.SWIPE_LEFT:
                    pyautogui.press('prevtrack')
                    self.last_action_time = current_time
                elif self.current_gesture == Gesture.SWIPE_RIGHT:
                    pyautogui.press('nexttrack')
                    self.last_action_time = current_time
                elif self.current_gesture == Gesture.PINCH:
                    pyautogui.press('playpause')
                    self.last_action_time = current_time
            
            elif self.control_mode == "keyboard":
                if self.current_gesture == Gesture.THUMB_UP:
                    pyautogui.press('volumeup')
                    self.last_action_time = current_time
                elif self.current_gesture == Gesture.THUMB_DOWN:
                    pyautogui.press('volumedown')
                    self.last_action_time = current_time
                elif self.current_gesture == Gesture.SWIPE_UP:
                    pyautogui.press('volumeup')
                    self.last_action_time = current_time
                elif self.current_gesture == Gesture.SWIPE_DOWN:
                    pyautogui.press('volumedown')
                    self.last_action_time = current_time
        
        except Exception as e:
            print(f"Error executing control: {e}")
    
    def _update_gesture_visualization(self):
        gesture_desc = {
            Gesture.NONE: "No active gesture detected",
            Gesture.POINTING: "🖐️ Pointing - Mouse movement",
            Gesture.PINCH: "🤏 Pinch - Left click",
            Gesture.THUMB_UP: "👍 Thumb up - Right click",
            Gesture.THUMB_DOWN: "👎 Thumb down - Volume down",
            Gesture.SWIPE_LEFT: "👈 Swipe left - Previous track",
            Gesture.SWIPE_RIGHT: "👉 Swipe right - Next track",
            Gesture.SWIPE_UP: "👆 Swipe up - Volume up",
            Gesture.SWIPE_DOWN: "👇 Swipe down - Volume down"
        }
        
        mode_desc = {
            "mouse": "Mouse control mode",
            "media": "Media control mode",
            "keyboard": "Keyboard shortcuts mode"
        }
        
        description = f"{mode_desc[self.control_mode]}\n\nCurrent gesture: {gesture_desc.get(self.current_gesture, 'Unknown')}"
        dpg.set_value(self.gesture_visual, description)
        dpg.set_value(self.gesture_text, f"Gesture: {self.current_gesture.name}")

if __name__ == "__main__":
    controller = GestureController()
    while controller.running:
        time.sleep(0.1)
