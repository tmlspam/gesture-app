import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time
import pyautogui
#kingtml

class Gesture(Enum):
    NONE = 0
    SWIPE_LEFT = 1
    SWIPE_RIGHT = 2
    SWIPE_UP = 3
    SWIPE_DOWN = 4
    PINCH = 5
    FIST = 6
    OPEN_PALM = 7
    POINTING = 8

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
        
        # FPS calculation
        self.prev_time = time.time()
        self.curr_time = time.time()
        self.fps = 0
        
        # Gesture detection parameters
        self.gesture_threshold = 3
        self.swipe_threshold = 0.15
        self.velocity_threshold = 0.5
        
        # Screen info for mouse control
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Mouse control smoothing
        self.mouse_positions = []
        self.smoothing_window = 3
        
        # Current state
        self.current_gesture = Gesture.NONE
        self.prev_landmarks = None
        self.control_mode = "mouse"  # mouse/keyboard/media

    def process_frame(self, frame):
        # Calculate FPS
        self.curr_time = time.time()
        self.fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Improves performance
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Reset frame
        output_frame = frame.copy()
        self.current_gesture = Gesture.NONE
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get and normalize landmarks
            current_landmarks = self._get_normalized_landmarks(hand_landmarks)
            
            # Detect gesture
            self.current_gesture = self._detect_gesture(current_landmarks)
            
            # Execute control action
            self._execute_control(current_landmarks)
            
            # Update previous landmarks
            self.prev_landmarks = current_landmarks
        
        # Display debug info
        self._draw_debug_info(output_frame)
        
        return output_frame, self.current_gesture

    def _get_normalized_landmarks(self, hand_landmarks):
        """Optimized landmark extraction"""
        lm = hand_landmarks.landmark
        return {
            'wrist': np.array([lm[0].x, lm[0].y]),
            'thumb_tip': np.array([lm[4].x, lm[4].y]),
            'index_tip': np.array([lm[8].x, lm[8].y]),
            'middle_tip': np.array([lm[12].x, lm[12].y]),
            'ring_tip': np.array([lm[16].x, lm[16].y]),
            'pinky_tip': np.array([lm[20].x, lm[20].y])
        }

    def _detect_gesture(self, current_landmarks):
        """Optimized gesture detection"""
        if self.prev_landmarks is None:
            return Gesture.NONE
        
        # Calculate movement
        dx = current_landmarks['wrist'][0] - self.prev_landmarks['wrist'][0]
        dy = current_landmarks['wrist'][1] - self.prev_landmarks['wrist'][1]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Detect swipes
        if velocity > self.velocity_threshold:
            if abs(dx) > abs(dy):  # Horizontal swipe
                return Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
            else:  # Vertical swipe
                return Gesture.SWIPE_DOWN if dy > 0 else Gesture.SWIPE_UP
        
        # Detect static gestures
        thumb_index_dist = np.linalg.norm(
            current_landmarks['thumb_tip'] - current_landmarks['index_tip'])
        
        if thumb_index_dist < 0.05:
            return Gesture.PINCH
        elif self._is_fist(current_landmarks):
            return Gesture.FIST
        elif self._is_open_palm(current_landmarks):
            return Gesture.OPEN_PALM
        elif self._is_pointing(current_landmarks):
            return Gesture.POINTING
            
        return Gesture.NONE

    def _is_fist(self, landmarks):
        """Optimized fist detection"""
        tips = [
            landmarks['thumb_tip'],
            landmarks['index_tip'],
            landmarks['middle_tip'],
            landmarks['ring_tip'],
            landmarks['pinky_tip']
        ]
        wrist = landmarks['wrist']
        avg_dist = np.mean([np.linalg.norm(tip - wrist) for tip in tips])
        return avg_dist < 0.15

    def _is_open_palm(self, landmarks):
        """Optimized open palm detection"""
        tips = [
            landmarks['thumb_tip'],
            landmarks['index_tip'],
            landmarks['middle_tip'],
            landmarks['ring_tip'],
            landmarks['pinky_tip']
        ]
        wrist = landmarks['wrist']
        avg_dist = np.mean([np.linalg.norm(tip - wrist) for tip in tips])
        return avg_dist > 0.25

    def _is_pointing(self, landmarks):
        """Optimized pointing detection"""
        index_tip = landmarks['index_tip']
        wrist = landmarks['wrist']
        index_extended = np.linalg.norm(index_tip - wrist) > 0.25
        
        other_tips = [
            landmarks['thumb_tip'],
            landmarks['middle_tip'],
            landmarks['ring_tip'],
            landmarks['pinky_tip']
        ]
        
        others_closed = all(np.linalg.norm(tip - wrist) < 0.15 for tip in other_tips)
        return index_extended and others_closed

    def _execute_control(self, landmarks):
        """Execute actions based on detected gesture"""
        if self.current_gesture == Gesture.NONE:
            return
            
        if self.control_mode == "mouse":
            self._control_mouse(landmarks)
        elif self.control_mode == "keyboard":
            self._control_keyboard()
        elif self.control_mode == "media":
            self._control_media()

    def _control_mouse(self, landmarks):
        """Smooth mouse control"""
        screen_x = int(landmarks['index_tip'][0] * self.screen_width)
        screen_y = int(landmarks['index_tip'][1] * self.screen_height)
        
        self.mouse_positions.append((screen_x, screen_y))
        if len(self.mouse_positions) > self.smoothing_window:
            self.mouse_positions.pop(0)
        
        avg_x = int(np.mean([pos[0] for pos in self.mouse_positions]))
        avg_y = int(np.mean([pos[1] for pos in self.mouse_positions]))
        
        pyautogui.moveTo(avg_x, avg_y, _pause=False)
        
        if self.current_gesture == Gesture.PINCH:
            pyautogui.click(_pause=False)
        elif self.current_gesture == Gesture.FIST:
            pyautogui.rightClick(_pause=False)

    def _control_keyboard(self):
        """Keyboard controls"""
        if self.current_gesture == Gesture.SWIPE_LEFT:
            pyautogui.press('left', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_RIGHT:
            pyautogui.press('right', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_UP:
            pyautogui.press('up', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_DOWN:
            pyautogui.press('down', _pause=False)

    def _control_media(self):
        """Media controls"""
        if self.current_gesture == Gesture.OPEN_PALM:
            pyautogui.press('playpause', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_RIGHT:
            pyautogui.press('nexttrack', _pause=False)
        elif self.current_gesture == Gesture.SWIPE_LEFT:
            pyautogui.press('prevtrack', _pause=False)

    def _draw_debug_info(self, frame):
        """Display debug information with fixed FPS counter"""
        # FPS counter
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current gesture
        cv2.putText(frame, f"Gesture: {self.current_gesture.name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Control mode
        cv2.putText(frame, f"Mode: {self.control_mode}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    controller = GestureController()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame, gesture = controller.process_frame(frame)
        
        # Display result
        cv2.imshow('Gesture Control', processed_frame)
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            controller.control_mode = "mouse"
        elif key == ord('k'):
            controller.control_mode = "keyboard"
        elif key == ord('a'):
            controller.control_mode = "media"
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()