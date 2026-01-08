"""
Video Proctoring Analyzer
Analyzes a video file for gaze direction and multiple face detection
Based on gaze_direction_detector.py but works with video files
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models/face_landmarker.task"
)

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

face_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = FaceLandmarker.create_from_options(
    face_landmarker_options
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

# Smoothing buffer
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
    """Estimate head rotation (yaw)"""
    nose = landmarks.landmark[NOSE_TIP]
    left_eye = landmarks.landmark[LEFT_EYE_OUTER]
    right_eye = landmarks.landmark[RIGHT_EYE_OUTER]
    left_mouth = landmarks.landmark[LEFT_MOUTH]
    right_mouth = landmarks.landmark[RIGHT_MOUTH]
    
    nose_pt = np.array([nose.x * img_w, nose.y * img_h])
    left_eye_pt = np.array([left_eye.x * img_w, left_eye.y * img_h])
    right_eye_pt = np.array([right_eye.x * img_w, right_eye.y * img_h])
    left_mouth_pt = np.array([left_mouth.x * img_w, left_mouth.y * img_h])
    right_mouth_pt = np.array([right_mouth.x * img_w, right_mouth.y * img_h])
    
    face_center_x = (left_eye_pt[0] + right_eye_pt[0]) / 2.0
    face_width = abs(right_eye_pt[0] - left_eye_pt[0])
    mouth_center_x = (left_mouth_pt[0] + right_mouth_pt[0]) / 2.0
    
    if face_width > 0:
        nose_offset = (nose_pt[0] - face_center_x) / face_width
        mouth_offset = (mouth_center_x - face_center_x) / face_width
        avg_offset = (nose_offset + mouth_offset) / 2.0
        head_yaw = avg_offset
    else:
        head_yaw = 0.0
    
    return head_yaw

def determine_gaze_direction(landmarks, img_w, img_h):
    """Determine gaze direction with head pose compensation"""
    head_yaw = estimate_head_pose(landmarks, img_w, img_h)
    
    left_iris_center = get_iris_center(landmarks, LEFT_IRIS, img_w, img_h)
    right_iris_center = get_iris_center(landmarks, RIGHT_IRIS, img_w, img_h)
    
    left_eye_center = get_eye_center(landmarks, LEFT_EYE, img_w, img_h)
    right_eye_center = get_eye_center(landmarks, RIGHT_EYE, img_w, img_h)
    
    left_iris_offset = left_iris_center - left_eye_center
    right_iris_offset = right_iris_center - right_eye_center
    
    avg_offset_x = (left_iris_offset[0] + right_iris_offset[0]) / 2.0
    avg_offset_y = (left_iris_offset[1] + right_iris_offset[1]) / 2.0
    
    face_width = abs(right_eye_center[0] - left_eye_center[0])
    
    if face_width > 0:
        head_rotation_pixel_offset = head_yaw * face_width * 0.4
        compensated_offset_x = avg_offset_x + head_rotation_pixel_offset
    else:
        compensated_offset_x = avg_offset_x
    
    if face_width > 0:
        HORIZONTAL_THRESHOLD = max(3.0, face_width * 0.03)
        VERTICAL_THRESHOLD = max(3.0, face_width * 0.03)
        HEAD_TURN_THRESHOLD = 0.15
    else:
        HORIZONTAL_THRESHOLD = 5.0
        VERTICAL_THRESHOLD = 5.0
        HEAD_TURN_THRESHOLD = 0.15
    
    if abs(head_yaw) > HEAD_TURN_THRESHOLD:
        movement_type = "HEAD"
    else:
        movement_type = "EYE"
    
    if compensated_offset_x < -HORIZONTAL_THRESHOLD:
        horizontal = "LEFT"
    elif compensated_offset_x > HORIZONTAL_THRESHOLD:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"
    
    if avg_offset_y < -VERTICAL_THRESHOLD:
        vertical = "UP"
    elif avg_offset_y > VERTICAL_THRESHOLD:
        vertical = "DOWN"
    else:
        vertical = "CENTER"
    
    if horizontal == "CENTER"and vertical == "CENTER":
        direction = "CENTER"
    elif horizontal == "CENTER":
        direction = vertical
    elif vertical == "CENTER":
        direction = horizontal
    else:
        direction = f"{vertical}_{horizontal}"
    
    gaze_history.append((direction, movement_type))
    
    if len(gaze_history) >= 3:
        direction_counts = {}
        for d, mt in gaze_history:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        smoothed_direction = max(direction_counts, key=direction_counts.get)
        for d, mt in reversed(gaze_history):
            if d == smoothed_direction:
                return smoothed_direction, mt
        return smoothed_direction, movement_type
    
    return direction, movement_type

class GazeTracker:
    """Track gaze statistics"""
    def __init__(self):
        self.start_time = None
        self.current_direction = None
        self.current_movement_type = None
        self.direction_start_time = None
        
        self.left_duration = 0.0
        self.right_duration = 0.0
        self.left_events = []
        self.right_events = []
        
    def update(self, direction, movement_type, current_time):
        """Update tracking with new gaze direction"""
        if self.start_time is None:
            self.start_time = current_time
            self.direction_start_time = current_time
        
        if direction != self.current_direction:
            if self.direction_start_time is not None and self.current_direction is not None:
                duration = current_time - self.direction_start_time
                
                if self.current_direction == "LEFT"or "LEFT"in self.current_direction:
                    self.left_duration += duration
                    self.left_events.append({
                        'timestamp': self.direction_start_time - self.start_time,
                        'duration': duration,
                        'movement_type': self.current_movement_type
                    })
                elif self.current_direction == "RIGHT"or "RIGHT"in self.current_direction:
                    self.right_duration += duration
                    self.right_events.append({
                        'timestamp': self.direction_start_time - self.start_time,
                        'duration': duration,
                        'movement_type': self.current_movement_type
                    })
            
            self.current_direction = direction
            self.current_movement_type = movement_type
            self.direction_start_time = current_time
        else:
            self.current_movement_type = movement_type
    
    def finalize(self, current_time):
        """Finalize tracking for the last direction"""
        if self.direction_start_time is not None and self.current_direction is not None:
            duration = current_time - self.direction_start_time
            
            if self.current_direction == "LEFT"or "LEFT"in self.current_direction:
                self.left_duration += duration
                self.left_events.append({
                    'timestamp': self.direction_start_time - self.start_time,
                    'duration': duration,
                    'movement_type': self.current_movement_type
                })
            elif self.current_direction == "RIGHT"or "RIGHT"in self.current_direction:
                self.right_duration += duration
                self.right_events.append({
                    'timestamp': self.direction_start_time - self.start_time,
                    'duration': duration,
                    'movement_type': self.current_movement_type
                })
    
    def get_statistics_text(self):
        """Get statistics as formatted text"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        lines = []
        lines.append("=" * 80)
        lines.append("GAZE TRACKING STATISTICS")
        lines.append("=" * 80)
        lines.append(f"Total tracking time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        lines.append(f"\nLEFT GAZE:")
        lines.append(f"Total duration: {self.left_duration:.2f} seconds ({self.left_duration/60:.2f} minutes)")
        lines.append(f"Percentage of time: {(self.left_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        lines.append(f"Number of events: {len(self.left_events)}")
        
        if self.left_events:
            lines.append(f"Events with timestamps:")
            for i, event in enumerate(self.left_events[:10], 1):  # Limit to first 10
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                lines.append(f"Event {i}: Time {timestamp_str}, Duration {duration_str}, Type: {event['movement_type']}")
            if len(self.left_events) > 10:
                lines.append(f"    ... and {len(self.left_events) - 10} more events")
        
        lines.append(f"\nRIGHT GAZE:")
        lines.append(f"Total duration: {self.right_duration:.2f} seconds ({self.right_duration/60:.2f} minutes)")
        lines.append(f"Percentage of time: {(self.right_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        lines.append(f"Number of events: {len(self.right_events)}")
        
        if self.right_events:
            lines.append(f"Events with timestamps:")
            for i, event in enumerate(self.right_events[:10], 1):
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                lines.append(f"Event {i}: Time {timestamp_str}, Duration {duration_str}, Type: {event['movement_type']}")
            if len(self.right_events) > 10:
                lines.append(f"    ... and {len(self.right_events) - 10} more events")
        
        left_head_count = sum(1 for e in self.left_events if e['movement_type'] == 'HEAD')
        left_eye_count = sum(1 for e in self.left_events if e['movement_type'] == 'EYE')
        right_head_count = sum(1 for e in self.right_events if e['movement_type'] == 'HEAD')
        right_eye_count = sum(1 for e in self.right_events if e['movement_type'] == 'EYE')
        
        lines.append(f"\nMOVEMENT TYPE BREAKDOWN:")
        lines.append(f"LEFT - Head turns: {left_head_count}, Eye movements: {left_eye_count}")
        lines.append(f"RIGHT - Head turns: {right_head_count}, Eye movements: {right_eye_count}")
        lines.append("=" * 80)
        
        return "\n".join(lines)

class MultipleFaceTracker:
    """Track multiple face detection periods"""
    def __init__(self):
        self.start_time = None
        self.is_tracking_multiple = False
        self.multiple_face_start_time = None
        self.multiple_face_events = []
        self.total_multiple_face_duration = 0.0
        self.current_max_faces = 0
        
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
        """Update the maximum number of faces"""
        if self.is_tracking_multiple:
            self.current_max_faces = max(self.current_max_faces, num_faces)
    
    def finalize(self, current_time):
        """Finalize tracking"""
        if self.is_tracking_multiple:
            self.end_multiple_face_period(current_time)
    
    def get_statistics_text(self):
        """Get statistics as formatted text"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("MULTIPLE FACE DETECTION STATISTICS")
        lines.append("=" * 80)
        lines.append(f"Total tracking time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        lines.append(f"\nMULTIPLE FACES DETECTED:")
        lines.append(f"Total duration: {self.total_multiple_face_duration:.2f} seconds ({self.total_multiple_face_duration/60:.2f} minutes)")
        lines.append(f"Percentage of time: {(self.total_multiple_face_duration/total_time*100) if total_time > 0 else 0:.2f}%")
        lines.append(f"Number of periods: {len(self.multiple_face_events)}")
        
        if self.multiple_face_events:
            lines.append(f"Periods with timestamps:")
            for i, event in enumerate(self.multiple_face_events[:10], 1):
                timestamp_str = f"{event['timestamp']:.2f}s"
                duration_str = f"{event['duration']:.2f}s"
                num_faces = event.get('num_faces', 0)
                lines.append(f"Period {i}: Time {timestamp_str}, Duration {duration_str}, Faces: {num_faces}")
            if len(self.multiple_face_events) > 10:
                lines.append(f"    ... and {len(self.multiple_face_events) - 10} more periods")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

def analyze_video_proctoring(video_path: str, output_txt_path: str = None) -> dict:
    """
    Analyze a video file for proctoring
    
    Args:
        video_path: Path to video file
        output_txt_path: Optional path to save TXT report
    
    Returns:
        dict with analysis results and report text
    """
    try:
        logger.info(f"Starting proctoring analysis of: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "report_text": None
            }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            error_msg = f"Could not open video file: {video_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "report_text": None
            }
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {fps} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Initialize trackers
        gaze_tracker = GazeTracker()
        multiple_face_tracker = MultipleFaceTracker()
        
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame for accuracy (you can skip frames if needed for speed)
            current_time = frame_count / fps if fps > 0 else 0
            
            h, w = frame.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            timestamp_ms = int(current_time * 1000)

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            # Perform face landmark detection
            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Get number of faces detected
            num_faces = len(result.face_landmarks)

            # Check for multiple faces
            if num_faces > 1:
                if not multiple_face_tracker.is_tracking_multiple:
                    multiple_face_tracker.start_multiple_face_period(num_faces, current_time)
                else:
                    multiple_face_tracker.update_multiple_face_count(num_faces)
            elif num_faces == 1:
                if multiple_face_tracker.is_tracking_multiple:
                    multiple_face_tracker.end_multiple_face_period(current_time)
                
                face_landmarks = result.face_landmarks[0]
                direction, movement_type = determine_gaze_direction(face_landmarks, w, h)
                gaze_tracker.update(direction, movement_type, current_time)
                processed_frames += 1
            else:
                if multiple_face_tracker.is_tracking_multiple:
                    multiple_face_tracker.end_multiple_face_period(current_time)
            
            # Log progress every 10%
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Finalize tracking
        final_time = duration
        gaze_tracker.finalize(final_time)
        multiple_face_tracker.finalize(final_time)
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VIDEO PROCTORING ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Video file: {os.path.basename(video_path)}")
        report_lines.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        report_lines.append(f"Total frames: {total_frames}")
        report_lines.append(f"Processed frames: {processed_frames}")
        report_lines.append(f"FPS: {fps}")
        report_lines.append("")
        
        # Add gaze statistics
        report_lines.append(gaze_tracker.get_statistics_text())
        report_lines.append("")
        
        # Add multiple face statistics
        report_lines.append(multiple_face_tracker.get_statistics_text())
        report_lines.append("")
        
        # Summary
        report_lines.append("=" * 80)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 80)
        
        # Calculate integrity score
        total_time = duration
        suspicious_time = gaze_tracker.left_duration + gaze_tracker.right_duration + multiple_face_tracker.total_multiple_face_duration
        integrity_percentage = ((total_time - suspicious_time) / total_time * 100) if total_time > 0 else 0
        
        report_lines.append(f"Integrity Score: {integrity_percentage:.2f}%")
        report_lines.append(f"Looking away time: {suspicious_time:.2f}s ({suspicious_time/60:.2f} min)")
        report_lines.append(f"Multiple faces detected: {len(multiple_face_tracker.multiple_face_events)} periods")
        report_lines.append("")
        
        # Warnings
        warnings = []
        if gaze_tracker.left_duration + gaze_tracker.right_duration > duration * 0.2:
            warnings.append("High amount of looking away detected (>20% of time)")
        if len(multiple_face_tracker.multiple_face_events) > 0:
            warnings.append(f"Multiple faces detected {len(multiple_face_tracker.multiple_face_events)} times")
        if len(gaze_tracker.left_events) + len(gaze_tracker.right_events) > 50:
            warnings.append("High frequency of gaze changes detected")
        
        if warnings:
            report_lines.append("WARNINGS:")
            for warning in warnings:
                report_lines.append(f"  {warning}")
        else:
            report_lines.append("No major violations detected")
        
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Processing time: {time.time() - start_time:.2f} seconds")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_txt_path:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_txt_path}")
        
        # Cleanup
        cap.release()
        
        logger.info("Proctoring analysis completed successfully")
        
        return {
            "success": True,
            "report_text": report_text,
            "output_file": output_txt_path,
            "video_path": video_path,
            "duration": duration,
            "integrity_score": integrity_percentage,
            "left_gaze_duration": gaze_tracker.left_duration,
            "right_gaze_duration": gaze_tracker.right_duration,
            "multiple_face_periods": len(multiple_face_tracker.multiple_face_events),
            "warnings": warnings
        }
        
    except Exception as e:
        error_msg = f"Error analyzing video: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "report_text": None
        }
    
face_landmarker.close()

if __name__ == "__main__":
    # Test with a video file
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = video_path.replace('.mp4', '_proctoring_report.txt')
        
        result = analyze_video_proctoring(video_path, output_path)
        
        if result['success']:
            print("\n" + result['report_text'])
            print(f"\n Report saved to: {output_path}")
        else:
            print(f"\n Error: {result['error']}")
    else:
        print("Usage: python video_proctoring_analyzer.py <video_file.mp4>")

