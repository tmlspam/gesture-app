import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import threading
import dearpygui.dearpygui as dpg
from enum import Enum, auto
from collections import deque
import queue
import json
import os
import logging
from threading import Lock

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
    TWO_FINGER_PINCH = auto()

class GestureController:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(threadName)s - %(message)s',
            filename='gesture_control.log'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing GestureController")
        
        # Thread synchronization
        self.lock = Lock()
        self.running = True
        
        # Initialize mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        # Screen and camera properties
        self.cam_width, self.cam_height = 640, 480
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Configuration with default values
        self.config = {
            'sensitivity': 0.7,
            'smoothing': 0.3,
            'gesture_threshold': 0.1,
            'swipe_velocity_threshold': 0.5,
            'pinch_threshold_multiplier': 0.08,
            'action_delay': 0.3,
            'min_gesture_confidence': 3
        }
        
        # State tracking
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.last_gesture_time = time.time()
        self.last_action_time = time.time()
        self.control_mode = "mouse"
        self.calibrating = False
        self.hand_size = 0.1
        self.gesture_confidence = {gesture: 0 for gesture in Gesture}
        self.last_mouse_pos = None
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.landmarks_queue = queue.Queue(maxsize=1)
        
        # Initialize GUI
        self.setup_gui()
        
        # Load configuration
        self.load_config()
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.run_capture, daemon=True, name="CaptureThread")
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True, name="ProcessingThread")
        self.capture_thread.start()
        self.processing_thread.start()
        
        # Disable PyAutoGUI failsafe
        pyautogui.FAILSAFE = False
    
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
                    self.camera_status = dpg.add_text("Camera: Initializing...", color=(255, 165, 0))
                
                # Control panel
                with dpg.child_window(width=400):
                    dpg.add_text("Gesture Control Pro", color=(0, 255, 255))
                    
                    # Settings section
                    with dpg.collapsing_header(label="Settings", default_open=True):
                        self.sensitivity_slider = dpg.add_slider_float(
                            label="Sensitivity", min_value=0.1, max_value=1.0,
                            default_value=self.config['sensitivity'], callback=self.update_config
                        )
                        self.smoothing_slider = dpg.add_slider_float(
                            label="Smoothing", min_value=0.0, max_value=1.0,
                            default_value=self.config['smoothing'], callback=self.update_config
                        )
                        self.gesture_threshold_slider = dpg.add_slider_float(
                            label="Gesture Threshold", min_value=0.01, max_value=0.2,
                            default_value=self.config['gesture_threshold'],
                            callback=self.update_config
                        )
                        dpg.add_button(label="Save Config", callback=self.save_config)
                    
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
    
    def update_config(self, sender=None, data=None):
        with self.lock:
            self.config['sensitivity'] = dpg.get_value(self.sensitivity_slider)
            self.config['smoothing'] = dpg.get_value(self.smoothing_slider)
            self.config['gesture_threshold'] = dpg.get_value(self.gesture_threshold_slider)
    
    def save_config(self):
        try:
            with open('gesture_config.json', 'w') as f:
                json.dump(self.config, f)
            dpg.set_value(self.calibration_text, "Configuration saved successfully")
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            dpg.set_value(self.calibration_text, f"Error saving config: {str(e)}")
    
    def load_config(self):
        try:
            if os.path.exists('gesture_config.json'):
                with open('gesture_config.json', 'r') as f:
                    loaded_config = json.load(f)
                    with self.lock:
                        self.config.update(loaded_config)
                    # Update GUI sliders
                    if dpg.does_item_exist(self.sensitivity_slider):
                        dpg.set_value(self.sensitivity_slider, self.config['sensitivity'])
                        dpg.set_value(self.smoothing_slider, self.config['smoothing'])
                        dpg.set_value(self.gesture_threshold_slider, self.config['gesture_threshold'])
                self.logger.info("Configuration loaded")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
    
    def start_calibration(self):
        with self.lock:
            self.calibrating = True
            self.calibration_step = 0
            self.calibration_values = []
        dpg.set_value(self.calibration_text, "Calibration: Show your open hand to the camera")
        self.logger.info("Starting calibration")
    
    def complete_calibration(self, hand_size):
        with self.lock:
            self.hand_size = hand_size
            self.calibrating = False
        dpg.set_value(self.calibration_text, f"Calibration complete. Hand size: {hand_size:.2f}")
        self.logger.info(f"Calibration complete. Hand size: {hand_size:.2f}")
    
    def set_control_mode(self, mode):
        with self.lock:
            self.control_mode = mode
        dpg.set_value(self.status_text, f"{mode.capitalize()} Mode")
        self._update_gesture_visualization()
        self.logger.info(f"Control mode changed to {mode}")
    
    def run_capture(self):
        self.logger.info("Starting capture thread")
        cap = None
        try:
            for camera_index in [0, 1, 2]:
                self.logger.info(f"Trying camera index {camera_index}")
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.logger.info(f"Camera opened at index {camera_index}")
                    dpg.set_value(self.camera_status, f"Camera: Using index {camera_index}")
                    break
            
            if not cap or not cap.isOpened():
                self.logger.error("No camera could be opened")
                dpg.set_value(self.camera_status, "Error: No camera found")
                self.running = False
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            prev_time = time.perf_counter()
            frame_count = 0
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue
                
                # Put frame in queue if not full
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Skip frame if queue is full
                
                # Calculate FPS every 10 frames
                frame_count += 1
                if frame_count % 10 == 0:
                    curr_time = time.perf_counter()
                    fps = 10 / (curr_time - prev_time)
                    dpg.set_value(self.fps_text, f"FPS: {int(fps)}")
                    prev_time = curr_time
                
                time.sleep(0.01)  # Prevent CPU overload
                
        except Exception as e:
            self.logger.error(f"Error in capture thread: {e}")
            dpg.set_value(self.camera_status, f"Camera Error: {str(e)}")
        finally:
            if cap and cap.isOpened():
                cap.release()
            self.logger.info("Capture thread stopped")
    
    def process_frames(self):
        self.logger.info("Starting processing thread")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Skip frames if queue is building up
                if self.frame_queue.qsize() > 1:
                    continue
                
                rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = {f"p{i}": (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}
                    
                    # Draw landmarks for visualization
                    self.mp_drawing.draw_landmarks(
                        rgb_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Put landmarks in queue for main thread
                    try:
                        self.landmarks_queue.put_nowait((rgb_frame, landmarks))
                    except queue.Full:
                        pass
                
                # Update camera feed
                resized_frame = cv2.resize(rgb_frame, (self.cam_width, self.cam_height)).astype(np.float32) / 255.0
                np.copyto(self.texture_data, resized_frame)
                dpg.set_value("camera_texture", self.texture_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
        
        self.logger.info("Processing thread stopped")
    
    def _handle_calibration(self, landmarks):
        if self.calibration_step == 0:
            wrist = landmarks['p0']
            middle_tip = landmarks['p12']
            hand_size = np.linalg.norm(np.array(middle_tip) - np.array(wrist))
            self.calibration_values.append(hand_size)
            
            if len(self.calibration_values) >= 10:
                avg_size = np.mean(self.calibration_values)
                self.complete_calibration(avg_size)
    
    def _detect_gesture(self, landmarks):
        try:
            thumb_tip = landmarks['p4']
            index_tip = landmarks['p8']
            middle_tip = landmarks['p12']
            wrist = landmarks['p0']
            
            # Dynamic threshold based on hand size
            pinch_threshold = 0.03 + (self.hand_size * self.config['pinch_threshold_multiplier'])
            
            # Calculate distances
            thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            thumb_middle_dist = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
            
            # Check pinch gestures first (high priority)
            if thumb_index_dist < pinch_threshold:
                self.gesture_confidence[Gesture.PINCH] += 1
                if self.gesture_confidence[Gesture.PINCH] > self.config['min_gesture_confidence']:
                    return Gesture.PINCH
            else:
                self.gesture_confidence[Gesture.PINCH] = 0
            
            # Check two-finger pinch (thumb + middle)
            if thumb_middle_dist < pinch_threshold:
                self.gesture_confidence[Gesture.TWO_FINGER_PINCH] += 1
                if self.gesture_confidence[Gesture.TWO_FINGER_PINCH] > self.config['min_gesture_confidence']:
                    return Gesture.TWO_FINGER_PINCH
            else:
                self.gesture_confidence[Gesture.TWO_FINGER_PINCH] = 0
            
            # Check pointing gesture
            index_extended = (index_tip[1] < landmarks['p6'][1]) and \
                            (index_tip[1] < landmarks['p10'][1]) and \
                            (index_tip[1] < landmarks['p14'][1])
            
            if index_extended:
                middle_closed = middle_tip[1] > landmarks['p10'][1]
                ring_closed = landmarks['p16'][1] > landmarks['p13'][1]
                pinky_closed = landmarks['p20'][1] > landmarks['p17'][1]
                
                if middle_closed and ring_closed and pinky_closed:
                    self.gesture_confidence[Gesture.POINTING] += 1
                    if self.gesture_confidence[Gesture.POINTING] > self.config['min_gesture_confidence']:
                        return Gesture.POINTING
            else:
                self.gesture_confidence[Gesture.POINTING] = 0
            
            # Check swipe gestures
            if self.prev_landmarks:
                dt = time.time() - self.last_gesture_time
                if dt > 0:
                    dx = landmarks['p8'][0] - self.prev_landmarks['p8'][0]
                    dy = landmarks['p8'][1] - self.prev_landmarks['p8'][1]
                    velocity_x = abs(dx) / dt
                    velocity_y = abs(dy) / dt
                    
                    if velocity_x > self.config['swipe_velocity_threshold'] and abs(dx) > self.config['gesture_threshold']:
                        return Gesture.SWIPE_LEFT if dx < 0 else Gesture.SWIPE_RIGHT
                    if velocity_y > self.config['swipe_velocity_threshold'] and abs(dy) > self.config['gesture_threshold']:
                        return Gesture.SWIPE_UP if dy < 0 else Gesture.SWIPE_DOWN
            
            # Check thumb gestures
            thumb_up = (thumb_tip[1] < landmarks['p3'][1]) and \
                      (thumb_tip[1] < landmarks['p2'][1])
            thumb_down = (thumb_tip[1] > landmarks['p3'][1]) and \
                        (thumb_tip[1] > landmarks['p2'][1])
            
            if thumb_up:
                self.gesture_confidence[Gesture.THUMB_UP] += 1
                if self.gesture_confidence[Gesture.THUMB_UP] > self.config['min_gesture_confidence']:
                    return Gesture.THUMB_UP
            else:
                self.gesture_confidence[Gesture.THUMB_UP] = 0
            
            if thumb_down:
                self.gesture_confidence[Gesture.THUMB_DOWN] += 1
                if self.gesture_confidence[Gesture.THUMB_DOWN] > self.config['min_gesture_confidence']:
                    return Gesture.THUMB_DOWN
            else:
                self.gesture_confidence[Gesture.THUMB_DOWN] = 0
            
            self.prev_landmarks = landmarks
            self.last_gesture_time = time.time()
            return Gesture.NONE
        
        except Exception as e:
            self.logger.error(f"Error in gesture detection: {e}")
            return Gesture.NONE
    
    def _execute_control(self, landmarks):
        current_time = time.time()
        if current_time - self.last_action_time < self.config['action_delay']:
            return
        
        try:
            if self.control_mode == "mouse":
                if self.current_gesture == Gesture.POINTING:
                    new_pos = [
                        landmarks['p8'][0] * self.screen_width * self.config['sensitivity'],
                        landmarks['p8'][1] * self.screen_height * self.config['sensitivity']
                    ]
                    
                    # Apply smoothing
                    if self.last_mouse_pos:
                        smoothed_pos = [
                            self.config['smoothing'] * new_pos[0] + (1 - self.config['smoothing']) * self.last_mouse_pos[0],
                            self.config['smoothing'] * new_pos[1] + (1 - self.config['smoothing']) * self.last_mouse_pos[1]
                        ]
                    else:
                        smoothed_pos = new_pos
                    
                    # Ensure position stays within screen bounds
                    smoothed_pos[0] = max(0, min(self.screen_width - 1, smoothed_pos[0]))
                    smoothed_pos[1] = max(0, min(self.screen_height - 1, smoothed_pos[1]))
                    
                    pyautogui.moveTo(int(smoothed_pos[0]), int(smoothed_pos[1]), _pause=False)
                    self.last_mouse_pos = smoothed_pos
                
                elif self.current_gesture == Gesture.PINCH:
                    pyautogui.click()
                    self.last_action_time = current_time
                    self.logger.debug("Mouse click executed")
                
                elif self.current_gesture == Gesture.TWO_FINGER_PINCH:
                    pyautogui.rightClick()
                    self.last_action_time = current_time
                    self.logger.debug("Right click executed")
            
            elif self.control_mode == "media":
                if self.current_gesture == Gesture.SWIPE_LEFT:
                    pyautogui.press('prevtrack')
                    self.last_action_time = current_time
                    self.logger.debug("Previous track command")
                elif self.current_gesture == Gesture.SWIPE_RIGHT:
                    pyautogui.press('nexttrack')
                    self.last_action_time = current_time
                    self.logger.debug("Next track command")
                elif self.current_gesture == Gesture.PINCH:
                    pyautogui.press('playpause')
                    self.last_action_time = current_time
                    self.logger.debug("Play/pause command")
                elif self.current_gesture == Gesture.THUMB_UP:
                    pyautogui.press('volumeup')
                    self.last_action_time = current_time
                    self.logger.debug("Volume up command")
                elif self.current_gesture == Gesture.THUMB_DOWN:
                    pyautogui.press('volumedown')
                    self.last_action_time = current_time
                    self.logger.debug("Volume down command")
            
            elif self.control_mode == "keyboard":
                if self.current_gesture == Gesture.THUMB_UP:
                    pyautogui.hotkey('ctrl', 'shift', 'right')
                    self.last_action_time = current_time
                    self.logger.debug("Next tab command")
                elif self.current_gesture == Gesture.THUMB_DOWN:
                    pyautogui.hotkey('ctrl', 'shift', 'left')
                    self.last_action_time = current_time
                    self.logger.debug("Previous tab command")
                elif self.current_gesture == Gesture.SWIPE_UP:
                    pyautogui.press('volumeup')
                    self.last_action_time = current_time
                    self.logger.debug("Volume up command")
                elif self.current_gesture == Gesture.SWIPE_DOWN:
                    pyautogui.press('volumedown')
                    self.last_action_time = current_time
                    self.logger.debug("Volume down command")
        
        except Exception as e:
            self.logger.error(f"Error executing control: {e}")
    
    def _update_gesture_visualization(self):
        gesture_desc = {
            Gesture.NONE: "No active gesture detected",
            Gesture.POINTING: "🖐️ Pointing - Mouse movement",
            Gesture.PINCH: "🤏 Pinch - Left click",
            Gesture.TWO_FINGER_PINCH: "🤌 Two-finger pinch - Right click",
            Gesture.THUMB_UP: "👍 Thumb up - Volume up/Next tab",
            Gesture.THUMB_DOWN: "👎 Thumb down - Volume down/Previous tab",
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

    def run(self):
        self.logger.info("Starting main application loop")
        try:
            while self.running and dpg.is_dearpygui_running():
                # Process landmarks if available
                try:
                    rgb_frame, landmarks = self.landmarks_queue.get_nowait()
                    if self.calibrating:
                        self._handle_calibration(landmarks)
                    else:
                        with self.lock:
                            self.current_gesture = self._detect_gesture(landmarks)
                            self._execute_control(landmarks)
                        self._update_gesture_visualization()
                except queue.Empty:
                    pass
                
                # Handle GUI events
                dpg.render_dearpygui_frame()
                
                # Check for exit condition
                if dpg.is_key_down(dpg.mvKey_Escape):
                    self.running = False
                
                time.sleep(0.01)  # Prevent CPU overload
            
        except Exception as e:
            self.logger.error(f"Main loop error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.logger.info("Cleaning up resources")
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        
        # Close MediaPipe resources
        self.hands.close()
        
        # Cleanup Dear PyGui
        if dpg.is_dearpygui_running():
            dpg.destroy_context()
        
        self.logger.info("Application shutdown complete")

if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except Exception as e:
        logging.error(f"Application crash: {str(e)}", exc_info=True)
