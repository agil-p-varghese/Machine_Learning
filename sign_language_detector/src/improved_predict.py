import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
from improved_preprocess import preprocess_live_image

# Load the trained model
model = load_model("saved_models/best_model.h5")
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Prediction smoothing
prediction_buffer = deque(maxlen=5)  # Store last 5 predictions
confidence_threshold = 0.7

def get_hand_bbox(landmarks, image_shape):
    h, w, _ = image_shape
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    
    # Add padding around hand
    padding = 20
    xmin = max(int(min(x_coords) * w) - padding, 0)
    xmax = min(int(max(x_coords) * w) + padding, w)
    ymin = max(int(min(y_coords) * h) - padding, 0)
    ymax = min(int(max(y_coords) * h) + padding, h)
    
    return xmin, ymin, xmax, ymax

def smooth_predictions(prediction_buffer):
    """Average predictions for stability"""
    if len(prediction_buffer) == 0:
        return None, 0
    
    # Average the prediction probabilities
    avg_pred = np.mean(prediction_buffer, axis=0)
    pred_index = np.argmax(avg_pred)
    confidence = avg_pred[pred_index]
    
    return pred_index, confidence

def draw_info_panel(frame, letter, confidence, fps):
    """Draw information panel on frame"""
    # Create info panel
    panel_height = 100
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    
    # Add text to panel
    cv2.putText(panel, f"Predicted Sign: {letter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(panel, f"Confidence: {confidence*100:.1f}%", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, f"FPS: {fps:.1f}", (400, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Add instructions
    cv2.putText(panel, "Press 'q' to quit, 'r' to reset", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Combine panel with frame
    return np.vstack([panel, frame])

# FPS calculation
fps_counter = 0
fps_start_time = cv2.getTickCount()

print("Starting sign detection...")
print("Make sure your hand is clearly visible and well-lit")
print("Press 'q' to quit, 'r' to reset predictions")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)
    
    current_letter = "None"
    current_confidence = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, 
                                 mp.solutions.hands.HAND_CONNECTIONS)
            
            # Get hand bounding box
            xmin, ymin, xmax, ymax = get_hand_bbox(hand_landmarks, frame.shape)
            
            # Extract and preprocess hand region
            hand_img = frame[ymin:ymax, xmin:xmax]
            processed_img = preprocess_live_image(hand_img)
            
            if processed_img is not None:
                # Make prediction
                pred = model.predict(processed_img, verbose=0)
                prediction_buffer.append(pred[0])
                
                # Get smoothed prediction
                pred_index, confidence = smooth_predictions(prediction_buffer)
                
                if pred_index is not None and confidence > confidence_threshold:
                    current_letter = labels[pred_index]
                    current_confidence = confidence
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > confidence_threshold else (0, 255, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Show prediction on hand
                cv2.putText(frame, f"{current_letter}", (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Calculate FPS
    fps_counter += 1
    if fps_counter >= 10:
        fps_end_time = cv2.getTickCount()
        fps = 10.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
        fps_start_time = fps_end_time
        fps_counter = 0
    else:
        fps = 0
    
    # Add information panel
    display_frame = draw_info_panel(frame, current_letter, current_confidence, fps)
    
    cv2.imshow("Enhanced Sign Detection", display_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        prediction_buffer.clear()
        print("Prediction buffer reset")

cap.release()
cv2.destroyAllWindows()
print("Sign detection stopped.")