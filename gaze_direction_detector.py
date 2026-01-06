"""
Gaze Direction Detector
A simple Python script to determine the direction you are looking using computer vision.
Uses MediaPipe for face mesh detection and iris tracking.
Now includes head pose compensation to handle head rotation.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Face mesh with iris landmarks - allow multiple faces
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,  # Allow up to 10 faces to be detected
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes and iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Key face landmarks for head pose estimation
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
CHIN = 175

# Smoothing buffer to reduce flickering
GAZE_HISTORY_SIZE = 5
gaze_history = deque(maxlen=GAZE_HISTORY_SIZE)

def get_iris_center(landmarks, iris_indices, img_w, img_h):
    """Calculate the center point of the iris"""
    iris_points = []
    for idx in iris_indices:
        x = landmarks.landmark[idx].x * img_w
        y = landmarks.landmark[idx].y * img_h
        iris_points.append([x, y])
    return np.mean(iris_points, axis=0)

def get_eye_center(landmarks, eye_indices, img_w, img_h):
    """Calculate the center point of the eye"""
    eye_points = []
    for idx in eye_indices:
        x = landmarks.landmark[idx].x * img_w
        y = landmarks.landmark[idx].y * img_h
        eye_points.append([x, y])
    return np.mean(eye_points, axis=0)

def estimate_head_pose(landmarks, img_w, img_h):
    """
    Estimate head rotation (yaw) to compensate for head turning.
    Uses multiple facial features for more robust estimation.
    Returns head_yaw in normalized units (positive = right, negative = left)
    """
    # Get key facial points
    nose = landmarks.landmark[NOSE_TIP]
    left_eye = landmarks.landmark[LEFT_EYE_OUTER]
    right_eye = landmarks.landmark[RIGHT_EYE_OUTER]
    left_mouth = landmarks.landmark[LEFT_MOUTH]
    right_mouth = landmarks.landmark[RIGHT_MOUTH]
    
    # Convert to pixel coordinates
    nose_pt = np.array([nose.x * img_w, nose.y * img_h])
    left_eye_pt = np.array([left_eye.x * img_w, left_eye.y * img_h])
    right_eye_pt = np.array([right_eye.x * img_w, right_eye.y * img_h])
    left_mouth_pt = np.array([left_mouth.x * img_w, left_mouth.y * img_h])
    right_mouth_pt = np.array([right_mouth.x * img_w, right_mouth.y * img_h])
    
    # Calculate face center and width using eyes
    face_center_x = (left_eye_pt[0] + right_eye_pt[0]) / 2.0
    face_width = abs(right_eye_pt[0] - left_eye_pt[0])
    
    # Also calculate mouth center for additional validation
    mouth_center_x = (left_mouth_pt[0] + right_mouth_pt[0]) / 2.0
    
    # Calculate head rotation using both nose and mouth offset
    if face_width > 0:
        # Nose offset (primary indicator)
        nose_offset = (nose_pt[0] - face_center_x) / face_width
        # Mouth offset (secondary validation)
        mouth_offset = (mouth_center_x - face_center_x) / face_width
        # Average for more stable estimation
        avg_offset = (nose_offset + mouth_offset) / 2.0
        # Normalize to -1 to 1 range (not degrees, just normalized value)
        head_yaw = avg_offset
    else:
        head_yaw = 0.0
    
    return head_yaw

def determine_gaze_direction(landmarks, img_w, img_h):
    """
    Determine gaze direction based on iris position relative to eye center.
    Now compensates for head rotation to prevent flickering when head turns.
    Returns: (direction string, movement_type)
    - direction: LEFT, RIGHT, CENTER, UP, DOWN, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
    - movement_type: "HEAD" if head is turned, "EYE" if only eyes moved
    """
    # Estimate head pose (yaw rotation)
    head_yaw = estimate_head_pose(landmarks, img_w, img_h)
    
    # Get iris centers
    left_iris_center = get_iris_center(landmarks, LEFT_IRIS, img_w, img_h)
    right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, img_w, img_h)
    
    # Get eye centers
    left_eye_center = get_eye_center(landmarks, LEFT_EYE, img_w, img_h)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE, img_w, img_h)
    
    # Calculate iris position relative to eye center
    left_iris_offset = left_iris_center - left_eye_center
    right_iris_offset = right_iris_center - right_eye_center
    
    # Average the offsets for both eyes
    avg_offset_x = (left_iris_offset[0] + right_iris_offset[0]) / 2.0
    avg_offset_y = (left_iris_offset[1] + right_iris_offset[1]) / 2.0
    
    # Calculate face width for compensation and thresholds
    face_width = abs(right_eye_center[0] - left_eye_center[0])
    
    # Compensate for head rotation
    # When head turns right (positive yaw), the entire face rotates right
    # The iris position relative to eye center changes due to head rotation
    # We need to compensate by adding the head rotation effect (inverted)
    # The compensation factor is tuned empirically - adjust if needed
    if face_width > 0:
        # Scale head_yaw (normalized -1 to 1) to pixel offset
        # When head turns right, iris appears to move left relative to eye center
        # So we add the head rotation offset to compensate
        head_rotation_pixel_offset = head_yaw * face_width * 0.4  # Tune 0.4 factor as needed
        compensated_offset_x = avg_offset_x + head_rotation_pixel_offset
    else:
        compensated_offset_x = avg_offset_x
    
    # Use adaptive thresholds based on face size for better accuracy
    if face_width > 0:
        # Normalize thresholds by face width
        HORIZONTAL_THRESHOLD = max(3.0, face_width * 0.03)  # 3% of face width or min 3px
        VERTICAL_THRESHOLD = max(3.0, face_width * 0.03)
        HEAD_TURN_THRESHOLD = 0.15  # Threshold for head turn detection (normalized)
    else:
        HORIZONTAL_THRESHOLD = 5.0
        VERTICAL_THRESHOLD = 5.0
        HEAD_TURN_THRESHOLD = 0.15
    
    # Determine if this is a head turn or eye movement
    # If head_yaw is significant, it's a head turn
    if abs(head_yaw) > HEAD_TURN_THRESHOLD:
        movement_type = "HEAD"
    else:
        movement_type = "EYE"
    
    # Determine horizontal direction (using compensated offset)
    if compensated_offset_x < -HORIZONTAL_THRESHOLD:
        horizontal = "LEFT"
    elif compensated_offset_x > HORIZONTAL_THRESHOLD:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"
    
    # Determine vertical direction
    if avg_offset_y < -VERTICAL_THRESHOLD:
        vertical = "UP"
    elif avg_offset_y > VERTICAL_THRESHOLD:
        vertical = "DOWN"
    else:
        vertical = "CENTER"
    
    # Combine directions
    if horizontal == "CENTER" and vertical == "CENTER":
        direction = "CENTER"
    elif horizontal == "CENTER":
        direction = vertical
    elif vertical == "CENTER":
        direction = horizontal
    else:
        direction = f"{vertical}_{horizontal}"
    
    # Add to history for smoothing
    gaze_history.append((direction, movement_type))
    
    # Return most common direction in recent history (reduces flickering)
    if len(gaze_history) >= 3:
        # Count occurrences of each direction
        direction_counts = {}
        for d, mt in gaze_history:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        # Return the most common direction
        smoothed_direction = max(direction_counts, key=direction_counts.get)
        # Get the movement type for the smoothed direction
        for d, mt in reversed(gaze_history):
            if d == smoothed_direction:
                return smoothed_direction, mt
        return smoothed_direction, movement_type
    
    return direction, movement_type

def draw_gaze_direction(frame, direction, movement_type, landmarks, img_w, img_h):
    """Draw visual indicators for gaze direction"""
    # Get iris centers for visualization
    left_iris_center = get_iris_center(landmarks, LEFT_IRIS, img_w, img_h)
    right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, img_w, img_h)
    
    # Draw iris centers
    cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 3, (0, 255, 0), -1)
    cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 3, (0, 255, 0), -1)
    
    # Display head yaw for debugging (optional)
    head_yaw = estimate_head_pose(landmarks, img_w, img_h)
    head_yaw_deg = head_yaw * 45.0  # Convert normalized to approximate degrees
    cv2.putText(frame, f"Head Yaw: {head_yaw_deg:.1f}°", (15, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Display movement type
    cv2.putText(frame, f"Movement: {movement_type}", (15, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Color coding for direction
    direction_colors = {
        "CENTER": (0, 255, 0),      # Green
        "LEFT": (0, 100, 255),      # Orange
        "RIGHT": (255, 100, 0),     # Orange
        "UP": (255, 0, 255),        # Magenta
        "DOWN": (0, 255, 255),      # Cyan
        "UP_LEFT": (255, 150, 0),   # Light Orange
        "UP_RIGHT": (255, 0, 150),  # Pink
        "DOWN_LEFT": (150, 255, 0), # Yellow-Green
        "DOWN_RIGHT": (0, 150, 255) # Light Blue
    }
    
    color = direction_colors.get(direction, (255, 255, 255))
    
    # Draw direction text with background
    text = f"Looking: {direction}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(frame, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (15, 35), font, font_scale, color, thickness)
    
    # Draw direction indicator arrows
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    arrow_length = 50
    if "LEFT" in direction:
        cv2.arrowedLine(frame, (center_x, center_y), 
                       (center_x - arrow_length, center_y), color, 3, tipLength=0.3)
    if "RIGHT" in direction:
        cv2.arrowedLine(frame, (center_x, center_y), 
                       (center_x + arrow_length, center_y), color, 3, tipLength=0.3)
    if "UP" in direction:
        cv2.arrowedLine(frame, (center_x, center_y), 
                       (center_x, center_y - arrow_length), color, 3, tipLength=0.3)
    if "DOWN" in direction:
        cv2.arrowedLine(frame, (center_x, center_y), 
                       (center_x, center_y + arrow_length), color, 3, tipLength=0.3)
    if direction == "CENTER":
        cv2.circle(frame, (center_x, center_y), 20, color, 3)

class GazeTracker:
    """Class to track gaze statistics"""
    def __init__(self):
        self.start_time = None
        self.current_direction = None
        self.current_movement_type = None
        self.direction_start_time = None
        
        # Statistics
        self.left_duration = 0.0
        self.right_duration = 0.0
        self.left_events = []  # List of (timestamp, duration, movement_type)
        self.right_events = []  # List of (timestamp, duration, movement_type)
        
    def update(self, direction, movement_type, current_time):
        """Update tracking with new gaze direction"""
        if self.start_time is None:
            self.start_time = current_time
            self.direction_start_time = current_time
        
        # Check if direction changed
        if direction != self.current_direction:
            # Calculate duration of previous direction
            if self.direction_start_time is not None and self.current_direction is not None:
                duration = current_time - self.direction_start_time
                
                # Track left/right durations and events
                if self.current_direction == "LEFT" or "LEFT" in self.current_direction:
                    self.left_duration += duration
                    self.left_events.append({
                        'timestamp': self.direction_start_time - self.start_time,
                        'duration': duration,
                        'movement_type': self.current_movement_type
                    })
                elif self.current_direction == "RIGHT" or "RIGHT" in self.current_direction:
                    self.right_duration += duration
                    self.right_events.append({
                        'timestamp': self.direction_start_time - self.start_time,
                        'duration': duration,
                        'movement_type': self.current_movement_type
                    })
            
            # Update to new direction
            self.current_direction = direction
            self.current_movement_type = movement_type
            self.direction_start_time = current_time
        else:
            # Same direction, just update movement type if it changed
            self.current_movement_type = movement_type
    
    def finalize(self, current_time):
        """Finalize tracking for the last direction"""
        if self.direction_start_time is not None and self.current_direction is not None:
            duration = current_time - self.direction_start_time
            
            if self.current_direction == "LEFT" or "LEFT" in self.current_direction:
                self.left_duration += duration
                self.left_events.append({
                    'timestamp': self.direction_start_time - self.start_time,
                    'duration': duration,
                    'movement_type': self.current_movement_type
                })
            elif self.current_direction == "RIGHT" or "RIGHT" in self.current_direction:
                self.right_duration += duration
                self.right_events.append({
                    'timestamp': self.direction_start_time - self.start_time,
                    'duration': duration,
                    'movement_type': self.current_movement_type
                })
    
    def print_statistics(self):
        """Print detailed statistics"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 80)
        print("GAZE TRACKING STATISTICS")
        print("=" * 80)
        print(f"Total tracking time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\nLEFT GAZE:")
        print(f"  Total duration: {self.left_duration:.2f} seconds ({self.left_duration/60:.2f} minutes)")
        print(f"  Percentage of time: {(self.left_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        print(f"  Number of events: {len(self.left_events)}")
        
        if self.left_events:
            print(f"  Events with timestamps:")
            for i, event in enumerate(self.left_events, 1):
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                print(f"    Event {i}: Time {timestamp_str}, Duration {duration_str}, Type: {event['movement_type']}")
        
        print(f"\nRIGHT GAZE:")
        print(f"  Total duration: {self.right_duration:.2f} seconds ({self.right_duration/60:.2f} minutes)")
        print(f"  Percentage of time: {(self.right_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        print(f"  Number of events: {len(self.right_events)}")
        
        if self.right_events:
            print(f"  Events with timestamps:")
            for i, event in enumerate(self.right_events, 1):
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                print(f"    Event {i}: Time {timestamp_str}, Duration {duration_str}, Type: {event['movement_type']}")
        
        # Summary by movement type
        left_head_count = sum(1 for e in self.left_events if e['movement_type'] == 'HEAD')
        left_eye_count = sum(1 for e in self.left_events if e['movement_type'] == 'EYE')
        right_head_count = sum(1 for e in self.right_events if e['movement_type'] == 'HEAD')
        right_eye_count = sum(1 for e in self.right_events if e['movement_type'] == 'EYE')
        
        print(f"\nMOVEMENT TYPE BREAKDOWN:")
        print(f"  LEFT - Head turns: {left_head_count}, Eye movements: {left_eye_count}")
        print(f"  RIGHT - Head turns: {right_head_count}, Eye movements: {right_eye_count}")
        
        print("=" * 80 + "\n")

class MultipleFaceTracker:
    """Class to track multiple face detection periods"""
    def __init__(self):
        self.start_time = None
        self.is_tracking_multiple = False
        self.multiple_face_start_time = None
        self.multiple_face_events = []  # List of (timestamp, duration, num_faces)
        self.total_multiple_face_duration = 0.0
        self.current_max_faces = 0  # Track max faces in current period
        
    def start_multiple_face_period(self, num_faces, current_time):
        """Start tracking a multiple face period"""
        if not self.is_tracking_multiple:
            self.is_tracking_multiple = True
            self.multiple_face_start_time = current_time
            self.current_max_faces = num_faces
            if self.start_time is None:
                self.start_time = current_time
    
    def end_multiple_face_period(self, current_time):
        """End tracking a multiple face period"""
        if self.is_tracking_multiple and self.multiple_face_start_time is not None:
            duration = current_time - self.multiple_face_start_time
            self.total_multiple_face_duration += duration
            
            self.multiple_face_events.append({
                'timestamp': self.multiple_face_start_time - (self.start_time or current_time),
                'duration': duration,
                'num_faces': self.current_max_faces
            })
            
            self.is_tracking_multiple = False
            self.multiple_face_start_time = None
            self.current_max_faces = 0
    
    def update_multiple_face_count(self, num_faces):
        """Update the maximum number of faces during a multiple face period"""
        if self.is_tracking_multiple:
            self.current_max_faces = max(self.current_max_faces, num_faces)
    
    def finalize(self, current_time):
        """Finalize tracking for the last multiple face period if still active"""
        if self.is_tracking_multiple:
            self.end_multiple_face_period(current_time)
    
    def print_statistics(self):
        """Print multiple face detection statistics"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 80)
        print("MULTIPLE FACE DETECTION STATISTICS")
        print("=" * 80)
        print(f"Total tracking time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"\nMULTIPLE FACES DETECTED:")
        print(f"  Total duration: {self.total_multiple_face_duration:.2f} seconds ({self.total_multiple_face_duration/60:.2f} minutes)")
        print(f"  Percentage of time: {(self.total_multiple_face_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        print(f"  Number of periods: {len(self.multiple_face_events)}")
        
        if self.multiple_face_events:
            print(f"  Periods with timestamps:")
            for i, event in enumerate(self.multiple_face_events, 1):
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                num_faces = event.get('num_faces', 0)
                print(f"    Period {i}: Time {timestamp_str}, Duration {duration_str}, Faces: {num_faces}")
        
        print("=" * 80 + "\n")

def main():
    """Main function to run the gaze direction detector"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize trackers
    gaze_tracker = GazeTracker()
    multiple_face_tracker = MultipleFaceTracker()
    
    print("=" * 60)
    print("Gaze Direction Detector with Multiple Face Detection")
    print("=" * 60)
    print("Press 'q' to quit and see statistics")
    print("Press 'r' to reset calibration (if needed)")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Get current time for tracking
        current_time = time.time()
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        # Count number of faces detected
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        
        # Check if multiple faces are detected
        if num_faces > 1:
            # Multiple faces detected - pause gaze tracking
            if not multiple_face_tracker.is_tracking_multiple:
                # Just started multiple face period
                multiple_face_tracker.start_multiple_face_period(num_faces, current_time)
            else:
                # Update max faces count
                multiple_face_tracker.update_multiple_face_count(num_faces)
            
            # Draw all detected faces
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )
            
            # Display multiple face warning
            text = f"MULTIPLE FACES DETECTED: {num_faces} faces"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(frame, (10, 10), (20 + text_width, 50 + text_height), (0, 0, 255), -1)
            
            # Draw text
            cv2.putText(frame, text, (15, 45), font, font_scale, (255, 255, 255), thickness)
            
            # Display status
            status_text = "Gaze tracking PAUSED"
            cv2.putText(frame, status_text, (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif num_faces == 1:
            # Single face - resume gaze tracking if we were tracking multiple
            if multiple_face_tracker.is_tracking_multiple:
                multiple_face_tracker.end_multiple_face_period(current_time)
            
            # Get the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh (optional, comment out if you want cleaner view)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                None,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # Determine gaze direction and movement type
            direction, movement_type = determine_gaze_direction(face_landmarks, w, h)
            
            # Update tracker (tracks all directions but only accumulates time for LEFT/RIGHT)
            gaze_tracker.update(direction, movement_type, current_time)
            
            # Draw gaze direction visualization
            draw_gaze_direction(frame, direction, movement_type, face_landmarks, w, h)
            
        else:
            # No face detected
            # End multiple face period if we were tracking one
            if multiple_face_tracker.is_tracking_multiple:
                multiple_face_tracker.end_multiple_face_period(current_time)
            
            cv2.putText(frame, "No Face Detected", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Gaze Direction Detector", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Reset requested (calibration reset not implemented in this version)")
    
    # Finalize tracking and print statistics
    final_time = time.time()
    gaze_tracker.finalize(final_time)
    multiple_face_tracker.finalize(final_time)
    
    # Print statistics
    gaze_tracker.print_statistics()
    multiple_face_tracker.print_statistics()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Gaze detector closed.")

if __name__ == "__main__":
    main()

