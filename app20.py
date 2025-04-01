import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Letter detection based on finger states
def detect_letter(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    mcp = [2, 5, 9, 13, 17]     # Finger base joints (for curl detection)
    
    # Check if each finger is extended (tip y-coordinate < base y-coordinate)
    extended = [
        hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcp[i]].y
        for i in range(5)
    ]
    
    # Letter detection logic
    if not extended[1] and not extended[2] and not extended[3] and not extended[4]:
        return "A"  # Only thumb out (fist with thumb up)
    elif extended[1] and extended[2] and extended[3] and extended[4]:
        return "B"  # All fingers extended
    elif extended[1] and extended[2] and not extended[3] and not extended[4]:
        return "C"  # Index and middle extended (like a 'C' shape)
    elif extended[1] and not extended[2] and not extended[3] and not extended[4]:
        return "D"  # Only index finger extended
    elif extended[0] and extended[1] and extended[2] and extended[3] and extended[4]:
        return "E"  # All fingers and thumb extended (open hand)
    else:
        return None

# Main capture loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Convert to RGB and process
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and detect letter
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            letter = detect_letter(hand_landmarks)
            if letter:
                cv2.putText(frame, f"Letter: {letter}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
