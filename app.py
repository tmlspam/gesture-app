import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

# Initialize MediaPipe Hands ok
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Control modes
MODES = ["mouse", "keyboard", "media"]
current_mode = 0  # Start with mouse mode

# Gesture thresholds
PINCH_THRESHOLD = 0.05
FIST_THRESHOLD = 0.15  # More reliable fist detection
SWIPE_THRESHOLD = 0.2

# Mouse smoothing
mouse_buffer = []
BUFFER_SIZE = 5

# FPS calculation
prev_time = 0

def get_gesture(landmarks):
    """Simplified gesture detection focusing on reliability"""
    # Get key points
    thumb = landmarks[4]
    index = landmarks[8]
    wrist = landmarks[0]
    
    # Calculate distances
    thumb_index_dist = np.linalg.norm(thumb - index)
    thumb_wrist_dist = np.linalg.norm(thumb - wrist)
    
    # Simple fist detection (thumb tucked in)
    if thumb_wrist_dist < FIST_THRESHOLD:
        return "fist"
    
    # Pinch detection
    if thumb_index_dist < PINCH_THRESHOLD:
        return "pinch"
    
    # Open hand (default state)
    return "open"

def process_frame(frame):
    global prev_time
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    gesture = None
    annotated_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmarks as numpy array
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(np.array([lm.x, lm.y]))
        landmarks = np.array(landmarks)
        
        # Detect gesture
        gesture = get_gesture(landmarks)
        
        # Control actions
        if gesture == "pinch":
            pyautogui.click()  # Left click
        elif gesture == "fist":
            pyautogui.rightClick()  # Right click
        
        # Mouse movement in mouse mode
        if MODES[current_mode] == "mouse":
            index_tip = landmarks[8]
            screen_x = int(index_tip[0] * screen_w)
            screen_y = int(index_tip[1] * screen_h)
            
            # Smooth mouse movement
            mouse_buffer.append((screen_x, screen_y))
            if len(mouse_buffer) > BUFFER_SIZE:
                mouse_buffer.pop(0)
            
            avg_x = int(np.mean([pos[0] for pos in mouse_buffer]))
            avg_y = int(np.mean([pos[1] for pos in mouse_buffer]))
            
            pyautogui.moveTo(avg_x, avg_y)
        
        # Draw landmarks (optional)
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
    
    # Display info
    cv2.putText(annotated_frame, f"Mode: {MODES[current_mode]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if gesture:
        cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame

def main():
    global current_mode
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame = process_frame(frame)
        
        # Display
        cv2.imshow('Gesture Control', processed_frame)
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):  # Cycle through modes
            current_mode = (current_mode + 1) % len(MODES)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
