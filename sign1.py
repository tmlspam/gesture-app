import cv2
import numpy as np
import time
import torch
import pyautogui
from enum import Enum

class Gesture(Enum):
    NONE = 0
    SIGN_A = 1
    SIGN_B = 2
    SIGN_C = 3
    SIGN_D = 4
    SIGN_E = 5

class SignLanguageInterpreter:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.6  # Confidence threshold
        self.prev_time = time.time()
        self.curr_time = time.time()
        self.fps = 0
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_gesture = Gesture.NONE
        self.gesture_confidence = 0
        self.running = True

    def process_frame(self, frame):
        self.curr_time = time.time()
        self.fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time

        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        detected_gesture = Gesture.NONE

        if not detections.empty:
            best_detection = detections.iloc[0]
            label = best_detection['name']
            detected_gesture = self._map_label_to_gesture(label)

        if detected_gesture == self.current_gesture:
            self.gesture_confidence = min(10, self.gesture_confidence + 1)
        else:
            self.gesture_confidence = max(0, self.gesture_confidence - 1)

        if self.gesture_confidence >= 3:
            self.current_gesture = detected_gesture
            self._execute_action()

        self._draw_debug_info(frame)
        return frame, self.current_gesture

    def _map_label_to_gesture(self, label):
        mapping = {
            'A': Gesture.SIGN_A,
            'B': Gesture.SIGN_B,
            'C': Gesture.SIGN_C,
            'D': Gesture.SIGN_D,
            'E': Gesture.SIGN_E
        }
        return mapping.get(label, Gesture.NONE)

    def _execute_action(self):
        if self.current_gesture == Gesture.SIGN_A:
            pyautogui.press('a', _pause=False)
        elif self.current_gesture == Gesture.SIGN_B:
            pyautogui.press('b', _pause=False)
        elif self.current_gesture == Gesture.SIGN_C:
            pyautogui.press('c', _pause=False)
        elif self.current_gesture == Gesture.SIGN_D:
            pyautogui.press('d', _pause=False)
        elif self.current_gesture == Gesture.SIGN_E:
            pyautogui.press('e', _pause=False)

    def _draw_debug_info(self, frame):
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture.name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def stop(self):
        self.running = False


def main():
    interpreter = SignLanguageInterpreter('yolov5_sign_language.pt')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened() and interpreter.running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        processed_frame, gesture = interpreter.process_frame(frame)
        cv2.imshow('Sign Language Interpreter', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Sign Language Interpreter', cv2.WND_PROP_VISIBLE) < 1:
            break

    interpreter.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
