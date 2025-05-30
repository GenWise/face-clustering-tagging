#!/usr/bin/env python3
# enhanced_video_processor.py - Complete video processing pipeline with content recognition

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import shutil

# Import our custom modules
from intelligent_frame_selector import IntelligentFrameSelector
from vision_api_processor import VisionAPIProcessor
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_video_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    """Complete video processing pipeline with face recognition and content analysis"""
    
    def __init__(self, config=None):
        """Initialize the video processor with configuration"""
        # Default configuration
        self.config = {
            'output_dir': 'processed_videos',
            'keyframes_dir': 'keyframes',
            'annotated_dir': 'annotated_keyframes',
            'detected_faces_dir': 'detected_faces',
            'face_db_dir': 'face_db',
            'max_api_calls_per_video': 60,
            'face_recognition_threshold': 0.6,
            'face_blur_threshold': 100,  # Laplacian variance threshold for blur detection
            'save_all_frames': False,
            'api_preference': ['openai', 'google', 'local'],
            'face_detector_name': 'buffalo_l',
            'face_detector_provider': 'CPUExecutionProvider'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Create output directories
        for dir_key in ['output_dir', 'keyframes_dir', 'annotated_dir', 'detected_faces_dir', 'face_db_dir']:
            dir_path = Path(self.config[dir_key])
            dir_path.mkdir(exist_ok=True, parents=True)
            
        # Initialize face detector
        self.face_detector = FaceAnalysis(
            name=self.config['face_detector_name'],
            providers=[self.config['face_detector_provider']]
        )
        self.face_detector.prepare(ctx_id=0)
        
        # Initialize frame selector
        frame_selector_config = {
            'max_api_calls_per_video': self.config['max_api_calls_per_video']
        }
        self.frame_selector = IntelligentFrameSelector(
            face_detector=self.face_detector,
            config=frame_selector_config
        )
        
        # Initialize vision API processor
        api_processor_config = {
            'api_preference': self.config['api_preference'],
            'cache_dir': str(Path(self.config['output_dir']) / 'vision_api_cache')
        }
        self.api_processor = VisionAPIProcessor(config=api_processor_config)
        
        # Load face database
        self.face_db = self._load_face_db()
        
    def _load_face_db(self):
        """Load face database with embeddings and metadata"""
        face_db = {
            'embeddings': {},
            'metadata': {}
        }
        
        # Load embeddings
        embeddings_path = Path(self.config['face_db_dir']) / 'face_embeddings.pkl'
        if embeddings_path.exists():
            try:
                import pickle
                with open(embeddings_path, 'rb') as f:
                    face_db['embeddings'] = pickle.load(f)
                logger.info(f"Loaded {len(face_db['embeddings'])} face embeddings from {embeddings_path}")
            except Exception as e:
                logger.error(f"Failed to load face embeddings: {e}")
                
        # Load metadata
        metadata_path = Path(self.config['face_db_dir']) / 'face_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_list = json.load(f)
                    
                # Convert list to dict for easier lookup
                for item in metadata_list:
                    face_id = item.get('face_id')
                    if face_id:
                        face_db['metadata'][face_id] = item
                        
                logger.info(f"Loaded metadata for {len(face_db['metadata'])} faces from {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to load face metadata: {e}")
                
        return face_db
        
    def is_blurry(self, image):
        """Detect if an image is blurry using Laplacian variance"""
        if image is None:
            return True
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return variance < self.config['face_blur_threshold']
        
    def match_face(self, face_embedding):
        """Match a face embedding against the database"""
        if not self.face_db['embeddings']:
            return None, 0.0
            
        best_match = None
        best_score = 0.0
        
        for face_id, db_embedding in self.face_db['embeddings'].items():
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, db_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(db_embedding)
            )
            
            # Convert to distance (lower is better)
            distance = 1.0 - similarity
            
            # Check if this is the best match so far
            if distance < (1.0 - self.config['face_recognition_threshold']) and similarity > best_score:
                best_match = face_id
                best_score = similarity
                
        # Get metadata for the best match
        metadata = None
        if best_match and best_match in self.face_db['metadata']:
            metadata = self.face_db['metadata'][best_match]
            
        return metadata, best_score
        
    def process_video(self, video_path):
        """Process a video with face recognition and content analysis"""
        video_path = Path(video_path)
        video_name = video_path.stem
        
        logger.info(f"Processing video: {video_path}")
        
        # Create video-specific output directories
        video_keyframes_dir = Path(self.config['keyframes_dir']) / video_name
        video_annotated_dir = Path(self.config['annotated_dir']) / video_name
        video_faces_dir = Path(self.config['detected_faces_dir']) / video_name
        
        for dir_path in [video_keyframes_dir, video_annotated_dir, video_faces_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
            
        # Step 1: Select keyframes
        keyframes_output_dir = video_keyframes_dir if self.config['save_all_frames'] else None
        selected_frames = self.frame_selector.process_video(
            video_path, 
            output_dir=keyframes_output_dir,
            save_frames=self.config['save_all_frames']
        )
        
        if not selected_frames:
            logger.error(f"No frames selected from {video_path}")
            return None
            
        logger.info(f"Selected {len(selected_frames)} keyframes from {video_path}")
        
        # Step 2: Process frames with Vision API
        vision_results = self.api_processor.process_video_frames(
            selected_frames,
            video_path=video_path,
            frames_dir=video_keyframes_dir
        )
        
        # Step 3: Process faces in each frame
        video_tags = {}  # Will store tags per frame
        all_video_tags = set()  # Will store unique tags across the video
        unmatched_face_count = 0
        
        for frame_info in vision_results:
            frame_index = frame_info.get('frame_index')
            if not frame_index:
                continue
                
            # Load the frame
            frame_path = None
            frame = None
            
            if 'filename' in frame_info:
                frame_path = video_keyframes_dir / frame_info['filename']
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
            
            if frame is None:
                # Extract frame from video
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    logger.error(f"Failed to extract frame {frame_index} from {video_path}")
                    continue
                    
                # Save frame for future use
                frame_path = video_keyframes_dir / f"{video_name}_frame{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
            # Detect faces in the frame
            try:
                faces = self.face_detector.get(frame)
                logger.info(f"Detected {len(faces)} faces in frame {frame_index}")
            except Exception as e:
                logger.error(f"Face detection error in frame {frame_index}: {e}")
                continue
                
            # Initialize frame tags
            frame_tags = {
                'tags': [],
                'confidence': {}
            }
            
            # Create a copy for annotation
            annotated_frame = frame.copy()
            
            # Process each face
            for i, face in enumerate(faces):
                # Extract face crop
                x1, y1, x2, y2 = map(int, face.bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                face_crop = frame[y1:y2, x1:x2]
                
                # Check if face is blurry
                if self.is_blurry(face_crop):
                    logger.info(f"Skipping blurry face in frame {frame_index}")
                    continue
                    
                # Match face against database
                metadata, confidence = self.match_face(face.embedding)
                
                if metadata and metadata.get('tag') and confidence >= self.config['face_recognition_threshold']:
                    # Face recognized
                    tag = metadata['tag']
                    
                    # Add tag to frame tags
                    if tag not in frame_tags['tags']:
                        frame_tags['tags'].append(tag)
                        frame_tags['confidence'][tag] = confidence
                        all_video_tags.add(tag)
                        
                    # Draw bounding box and label on annotated frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame, 
                        f"{tag} ({confidence:.2f})", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        1
                    )
                else:
                    # Unrecognized face
                    unmatched_face_count += 1
                    face_id = f"{video_name}_frame{frame_index}_face{i}"
                    
                    # Save face crop for later tagging
                    face_crop_path = video_faces_dir / f"{face_id}.jpg"
                    cv2.imwrite(str(face_crop_path), face_crop)
                    
                    # Also copy to root detected_faces directory for UI use
                    root_face_path = Path(self.config['detected_faces_dir']) / f"{face_id}.jpg"
                    shutil.copy(str(face_crop_path), str(root_face_path))
                    
                    # Draw bounding box on annotated frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated_frame, 
                        "Unknown", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255), 
                        1
                    )
            
            # Save annotated frame
            annotated_path = video_annotated_dir / f"{video_name}_frame{frame_index:06d}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_frame)
            
            # Add frame tags to video tags
            if frame_tags['tags']:
                video_tags[str(frame_index)] = frame_tags
                
        # Step 4: Combine with vision API results to create a comprehensive summary
        vision_summary = self.api_processor.summarize_video_content(vision_results)
        
        # Combine face recognition and vision results
        summary = {
            "video_path": str(video_path),
            "processed_at": datetime.now().isoformat(),
            "face_recognition": {
                "recognized_faces": list(all_video_tags),
                "unmatched_faces": unmatched_face_count
            },
            "vision_analysis": vision_summary,
            "frame_count": len(selected_frames),
            "keyframes_dir": str(video_keyframes_dir),
            "annotated_dir": str(video_annotated_dir),
            "faces_dir": str(video_faces_dir)
        }
        
        # Save results
        results_path = Path(self.config['output_dir']) / f"{video_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Update video tags file
        video_tags_path = Path("video_tags.json")
        all_video_tags_dict = {}
        
        if video_tags_path.exists():
            try:
                with open(video_tags_path, 'r') as f:
                    all_video_tags_dict = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load video tags: {e}")
                
        all_video_tags_dict[video_name] = video_tags
        
        with open(video_tags_path, 'w') as f:
            json.dump(all_video_tags_dict, f, indent=2)
            
        # Update unique tags by video file
        unique_tags_path = Path("unique_tags_by_video.json")
        unique_tags_dict = {}
        
        if unique_tags_path.exists():
            try:
                with open(unique_tags_path, 'r') as f:
                    unique_tags_dict = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load unique tags: {e}")
                
        unique_tags_dict[video_name] = list(all_video_tags)
        
        with open(unique_tags_path, 'w') as f:
            json.dump(unique_tags_dict, f, indent=2)
            
        logger.info(f"Video processing complete: {video_path}")
        logger.info(f"Recognized {len(all_video_tags)} unique faces")
        logger.info(f"Found {unmatched_face_count} unmatched faces")
        
        return summary

def main():
    """Run the enhanced video processor on a video file"""
    parser = argparse.ArgumentParser(description='Enhanced video processing with face recognition and content analysis')
    parser.add_argument('video_path', help='Path to the video file to process')
    parser.add_argument('--output-dir', default='processed_videos', help='Output directory for results')
    parser.add_argument('--keyframes-dir', default='keyframes', help='Directory for extracted keyframes')
    parser.add_argument('--annotated-dir', default='annotated_keyframes', help='Directory for annotated frames')
    parser.add_argument('--detected-faces-dir', default='detected_faces', help='Directory for detected faces')
    parser.add_argument('--face-db-dir', default='face_db', help='Directory for face database')
    parser.add_argument('--max-api-calls', type=int, default=60, help='Maximum API calls per video')
    parser.add_argument('--face-threshold', type=float, default=0.6, help='Face recognition threshold')
    parser.add_argument('--save-all-frames', action='store_true', help='Save all extracted frames')
    parser.add_argument('--api', choices=['openai', 'google', 'local'], help='Preferred API to use')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--google-key', help='Google API key')
    args = parser.parse_args()
    
    # Set up configuration
    config = {
        'output_dir': args.output_dir,
        'keyframes_dir': args.keyframes_dir,
        'annotated_dir': args.annotated_dir,
        'detected_faces_dir': args.detected_faces_dir,
        'face_db_dir': args.face_db_dir,
        'max_api_calls_per_video': args.max_api_calls,
        'face_recognition_threshold': args.face_threshold,
        'save_all_frames': args.save_all_frames
    }
    
    # Set API preference if specified
    if args.api:
        config['api_preference'] = [args.api]
        
    # Set API keys in environment variables
    if args.openai_key:
        os.environ['OPENAI_API_KEY'] = args.openai_key
    if args.google_key:
        os.environ['GOOGLE_API_KEY'] = args.google_key
        
    # Create and run processor
    processor = EnhancedVideoProcessor(config)
    
    # Process video
    try:
        result = processor.process_video(args.video_path)
        if result:
            print(f"Video processing complete. Results saved to {config['output_dir']}")
        else:
            print("Video processing failed.")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        print(f"Error processing video: {e}")
        
if __name__ == "__main__":
    main() 