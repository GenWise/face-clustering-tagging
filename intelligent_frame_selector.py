#!/usr/bin/env python3
# intelligent_frame_selector.py - Selects keyframes for advanced vision processing

import cv2
import numpy as np
from pathlib import Path
import time
import logging
from skimage.metrics import structural_similarity as ssim
from insightface.app import FaceAnalysis
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
FORCE_INTERVAL = 300  # Force API call every N frames (e.g., every 10 seconds at 30fps)
DIVERSITY_THRESHOLD = 0.25  # Minimum visual difference to consider a new frame
FACE_CHANGE_THRESHOLD = 1  # Minimum change in face count to trigger API call
MOTION_THRESHOLD = 0.15  # Minimum motion intensity to consider a new frame
MAX_API_CALLS_PER_VIDEO = 60  # Maximum API calls per video (adjust based on budget)
SCENE_CHANGE_THRESHOLD = 30.0  # Threshold for scene change detection

class IntelligentFrameSelector:
    """Selects informative keyframes for advanced vision processing"""
    
    def __init__(self, face_detector=None, config=None):
        """Initialize the frame selector with optional custom configuration"""
        # Load default configuration or use provided config
        self.config = {
            'force_interval': FORCE_INTERVAL,
            'diversity_threshold': DIVERSITY_THRESHOLD,
            'face_change_threshold': FACE_CHANGE_THRESHOLD,
            'motion_threshold': MOTION_THRESHOLD,
            'max_api_calls_per_video': MAX_API_CALLS_PER_VIDEO,
            'scene_change_threshold': SCENE_CHANGE_THRESHOLD
        }
        
        if config:
            self.config.update(config)
            
        # Initialize face detector if not provided
        if face_detector is None:
            self.face_detector = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=0)
        else:
            self.face_detector = face_detector
            
        # Initialize state variables
        self.reset_state()
        
    def reset_state(self):
        """Reset the state for a new video"""
        self.previous_frames = []  # Store last few frames for comparison
        self.previous_gray = None  # Previous grayscale frame for motion detection
        self.frame_count = 0
        self.api_calls = 0
        self.last_face_count = 0
        self.last_api_frame = 0
        self.selected_frames = []
        
    def calculate_visual_diversity(self, current_frame, previous_frame):
        """Calculate visual diversity between two frames using SSIM"""
        # Convert to grayscale for comparison
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            
        if len(previous_frame.shape) == 3:
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        else:
            previous_gray = previous_frame
        
        # Resize if needed to match dimensions
        if current_gray.shape != previous_gray.shape:
            previous_gray = cv2.resize(previous_gray, (current_gray.shape[1], current_gray.shape[0]))
        
        # Calculate SSIM (returns a value between -1 and 1, where 1 means identical)
        similarity = ssim(current_gray, previous_gray)
        
        # Convert to diversity score (0 to 1, where 1 is completely different)
        diversity = 1.0 - similarity
        
        return diversity
        
    def detect_faces_count(self, frame):
        """Count faces in the frame using InsightFace"""
        try:
            faces = self.face_detector.get(frame)
            return len(faces)
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return 0
            
    def calculate_motion_intensity(self, current_frame, previous_gray):
        """Calculate motion intensity using optical flow"""
        if previous_gray is None:
            if len(current_frame.shape) == 3:
                return 0, cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return 0, current_frame
            
        # Convert current frame to grayscale
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate motion magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_intensity = np.mean(magnitude)
        
        return motion_intensity, current_gray
        
    def detect_scene_change(self, current_frame, previous_frame):
        """Detect if there's a scene change between frames using histogram comparison"""
        # Convert to HSV color space
        if len(current_frame.shape) == 3:
            hsv_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            hsv_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
        else:
            # If grayscale, we can't convert to HSV
            return False
            
        # Calculate histograms
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges
        channels = [0, 1]
        
        hist_current = cv2.calcHist([hsv_current], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist_current, hist_current, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_previous = cv2.calcHist([hsv_previous], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist_previous, hist_previous, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms
        diff = cv2.compareHist(hist_current, hist_previous, cv2.HISTCMP_CHISQR)
        
        return diff > self.config['scene_change_threshold']
        
    def should_select_frame(self, frame):
        """Determine if the current frame should be selected for API processing"""
        self.frame_count += 1
        
        # Force selection at regular intervals or for first frame
        if self.frame_count == 1 or (self.frame_count - self.last_api_frame) >= self.config['force_interval']:
            self.last_api_frame = self.frame_count
            self.api_calls += 1
            
            # Store frame for future comparison
            if len(self.previous_frames) >= 5:  # Keep only last 5 frames
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            
            # Update motion detection state
            if len(frame.shape) == 3:
                self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                self.previous_gray = frame.copy()
                
            return True
            
        # Check if we've reached the API call budget
        if self.api_calls >= self.config['max_api_calls_per_video']:
            return False
            
        # Skip if no previous frames to compare with
        if not self.previous_frames:
            return False
            
        # Calculate visual diversity
        diversity_score = self.calculate_visual_diversity(frame, self.previous_frames[-1])
        
        # Calculate motion intensity
        motion_intensity, current_gray = self.calculate_motion_intensity(frame, self.previous_gray)
        self.previous_gray = current_gray
        
        # Count faces
        current_faces = self.detect_faces_count(frame)
        face_change = abs(current_faces - self.last_face_count) >= self.config['face_change_threshold']
        self.last_face_count = current_faces
        
        # Check for scene change
        is_scene_change = False
        if len(self.previous_frames) > 0:
            is_scene_change = self.detect_scene_change(frame, self.previous_frames[-1])
            
        # Make decision based on combined factors
        should_select = (
            diversity_score > self.config['diversity_threshold'] or
            motion_intensity > self.config['motion_threshold'] or
            face_change or
            is_scene_change
        )
        
        if should_select:
            self.api_calls += 1
            self.last_api_frame = self.frame_count
            
            # Store frame for future comparison
            if len(self.previous_frames) >= 5:  # Keep only last 5 frames
                self.previous_frames.pop(0)
            self.previous_frames.append(frame.copy())
            
            logger.info(f"Frame {self.frame_count} selected: diversity={diversity_score:.2f}, "
                      f"motion={motion_intensity:.2f}, faces={current_faces}, "
                      f"scene_change={is_scene_change}")
                      
        return should_select
        
    def process_video(self, video_path, output_dir=None, save_frames=False, frame_interval=30):
        """Process a video and select keyframes
        
        Args:
            video_path (str): Path to the video file
            output_dir (str, optional): Directory to save keyframes to
            save_frames (bool, optional): Whether to save all processed frames
            frame_interval (int, optional): Interval between frames to process (higher = fewer frames)
            
        Returns:
            List of selected frame information
        """
        video_path = Path(video_path)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {video_path.name}, {fps:.2f} fps, {duration:.2f} seconds")
        
        # Create output directory if needed
        if output_dir and save_frames:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
        # Reset state
        self.reset_state()
        
        # Prepare for frame processing
        selected_frames = []
        prev_frame = None
        prev_gray = None
        frame_index = 0
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_index += 1
            
            # Only process every n-th frame
            if frame_index % frame_interval != 0:
                continue
                
            # Check if this frame should be selected
            if self.should_select_frame(frame):
                timestamp = frame_index / fps if fps > 0 else 0
                frame_info = {
                    'frame_index': frame_index,
                    'timestamp': timestamp,
                    'timestamp_formatted': time.strftime('%H:%M:%S', time.gmtime(timestamp)),
                    'face_count': self.last_face_count
                }
                selected_frames.append(frame_info)
                
                # Save frame if requested
                if save_frames and output_dir:
                    frame_filename = f"{video_path.stem}_frame{frame_index:06d}.jpg"
                    cv2.imwrite(str(output_dir / frame_filename), frame)
                    frame_info['filename'] = frame_filename
                    
        cap.release()
        
        # Save selection metadata
        if output_dir:
            metadata = {
                'video_path': str(video_path),
                'total_frames': frame_count,
                'duration': duration,
                'fps': fps,
                'frames_processed': self.frame_count,
                'frames_selected': len(selected_frames),
                'selection_rate': len(selected_frames) / self.frame_count if self.frame_count > 0 else 0,
                'selected_frames': selected_frames
            }
            
            with open(output_dir / f"{video_path.stem}_selection.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Video processing complete: {len(selected_frames)} frames selected "
                  f"out of {self.frame_count} processed ({frame_count} total)")
                  
        return selected_frames

def main():
    """Run frame selection on a sample video"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intelligent keyframe selection for video processing')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', '-o', help='Output directory for selected frames', default='selected_frames')
    parser.add_argument('--save-frames', '-s', action='store_true', help='Save selected frames as images')
    parser.add_argument('--max-api-calls', '-m', type=int, help='Maximum API calls per video', default=MAX_API_CALLS_PER_VIDEO)
    args = parser.parse_args()
    
    # Create frame selector
    config = {'max_api_calls_per_video': args.max_api_calls}
    selector = IntelligentFrameSelector(config=config)
    
    # Process video
    selected_frames = selector.process_video(args.video_path, args.output, args.save_frames)
    
    print(f"Selected {len(selected_frames)} frames for API processing")
    
if __name__ == "__main__":
    main() 