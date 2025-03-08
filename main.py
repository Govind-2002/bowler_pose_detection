import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_pose = mp.solutions.pose

class BowlingAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        self.angle_buffer = deque(maxlen=15)
        self.wrist_pos_buffer = deque(maxlen=10)
        self.crease_y = None
        self.alpha = 0.3
        self.smoothed_angle = 0.0
        self.last_update_time = time.time()
        self.release_angle = None
        self.release_status = None
        self.release_no_ball = False
        self.frame_count = 0  # Added for debugging

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def detect_release_point(self, current_wrist_y):
        if len(self.wrist_pos_buffer) < 5:
            return False
        return current_wrist_y > np.mean(self.wrist_pos_buffer) and current_wrist_y == max(self.wrist_pos_buffer)

    def detect_no_ball(self, landmarks, frame_height):
        if self.crease_y is None:
            self.crease_y = frame_height * 0.85
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        return (right_foot.y * frame_height > self.crease_y or 
                left_foot.y * frame_height > self.crease_y)

    def draw_angle_analysis(self, frame, shoulder, elbow, wrist):
        angle = self.calculate_angle(shoulder, elbow, wrist)
        cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, elbow)), (0, 165, 255), 4)
        cv2.line(frame, tuple(map(int, elbow)), tuple(map(int, wrist)), (0, 165, 255), 4)
        cv2.circle(frame, tuple(map(int, shoulder)), 8, (0, 255, 0), -1)
        cv2.circle(frame, tuple(map(int, elbow)), 8, (0, 255, 255), -1)
        cv2.circle(frame, tuple(map(int, wrist)), 8, (0, 255, 0), -1)
        
        angle_color = (0, 255, 0) if angle >= 165 else (0, 0, 255)
        cv2.putText(frame, f"{int(angle)}°", 
                   (int(elbow[0]) + 15, int(elbow[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, angle_color, 2)
        
        start_angle = int(np.degrees(np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0])))
        end_angle = int(np.degrees(np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0])))
        cv2.ellipse(frame, tuple(map(int, elbow)), (30, 30), 
                   0, start_angle, end_angle, angle_color, 2)
        return angle, frame

    def draw_stats(self, frame):
        h, w = frame.shape[:2]
        stats_box = np.zeros((220, 320, 3), dtype=np.uint8)
        stats_box[:] = (45, 45, 45)
        
        angle_display = f"{int(self.smoothed_angle)}°" if self.smoothed_angle else "Initializing..."
        cv2.putText(stats_box, "LIVE BOWLING ANALYSIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(stats_box, f"Current Angle: {angle_display}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(stats_box, f"Legal Threshold: 165°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        frame[20:240, 20:340] = cv2.addWeighted(frame[20:240, 20:340], 0.3, stats_box, 0.7, 0)
        return frame

    def analyze_frame(self, frame):
        try:
            self.frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            angle = None
            
            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

                angle, frame = self.draw_angle_analysis(frame, shoulder, elbow, wrist)
                self.angle_buffer.append(angle)
                self.smoothed_angle = self.alpha * angle + (1 - self.alpha) * self.smoothed_angle

                current_wrist_y = wrist[1]
                self.wrist_pos_buffer.append(current_wrist_y)
                
                if self.detect_release_point(current_wrist_y):
                    no_ball = self.detect_no_ball(landmarks, h)
                    self.release_angle = self.smoothed_angle
                    self.release_status = "LEGAL" if self.release_angle >= 165 else "ILLEGAL"
                    self.release_no_ball = no_ball

                    color = (0, 255, 0) if self.release_status == "LEGAL" else (0, 0, 255)
                    cv2.putText(frame, f"{self.release_status} DELIVERY", (w//2-100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if self.release_no_ball:
                        cv2.putText(frame, "NO BALL!", (w//2-80, h//2+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame = self.draw_stats(frame)
            return frame
        
        except Exception as e:
            print(f"Error in frame {self.frame_count}: {str(e)}")
            return frame

    def get_final_result(self):
        return {
            'release_angle': self.release_angle,
            'release_status': self.release_status,
            'release_no_ball': self.release_no_ball
        }

def analyze_video(source):
    analyzer = BowlingAnalyzer()
    cap = cv2.VideoCapture(source)
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            analyzer.analyze_frame(frame)
    finally:
        cap.release()
    
    return analyzer.get_final_result()