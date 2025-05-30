#!/usr/bin/env python3
# vision_api_processor.py - Process selected frames with Vision APIs

import os
import cv2
import base64
import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VisionAPIProcessor:
    """Process frames with various Vision APIs for content recognition"""
    
    def __init__(self, api_key=None, config=None):
        """Initialize the API processor with keys and configuration"""
        # Default configuration
        self.config = {
            'api_preference': ['openai', 'google', 'local'],  # Order of API preference
            'cache_results': True,  # Cache API results to avoid duplicate calls
            'cache_dir': 'vision_api_cache',  # Directory to store cached results
            'retry_count': 2,  # Number of retries for failed API calls
            'retry_delay': 3,  # Delay between retries in seconds
            'confidence_threshold': 0.6,  # Minimum confidence for detections
            'local_model_path': None,  # Path to local model if available
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Set API keys
        self.api_keys = {}
        
        # Process the api_key parameter
        if api_key:
            if isinstance(api_key, dict):
                self.api_keys.update(api_key)
            else:
                # Try to guess which API the key is for
                if api_key.startswith('sk-'):
                    self.api_keys['openai'] = api_key
                elif len(api_key) > 30:
                    self.api_keys['google'] = api_key
        
        # Also check environment variables for API keys if not already set
        if 'openai' not in self.api_keys and 'OPENAI_API_KEY' in os.environ:
            self.api_keys['openai'] = os.environ['OPENAI_API_KEY']
            print(f"Using OpenAI API key from environment: {self.api_keys['openai'][:10]}...")
            
        if 'google' not in self.api_keys and 'GOOGLE_API_KEY' in os.environ:
            self.api_keys['google'] = os.environ['GOOGLE_API_KEY']
            print(f"Using Google API key from environment: {self.api_keys['google'][:10]}...")
            
        # Create cache directory if needed
        if self.config['cache_results']:
            cache_dir = Path(self.config['cache_dir'])
            cache_dir.mkdir(exist_ok=True, parents=True)
            
        # Initialize local model if specified
        self.local_model = None
        if 'local' in self.config['api_preference'] and self.config['local_model_path']:
            try:
                import torch
                from PIL import Image
                
                self.local_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                path=self.config['local_model_path'])
                logger.info(f"Loaded local model from {self.config['local_model_path']}")
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
                
    def _get_cache_path(self, image_path, api_name):
        """Generate a cache file path for the given image and API"""
        if not self.config['cache_results']:
            return None
            
        # Create a unique filename based on the image path and API
        image_path = Path(image_path)
        cache_filename = f"{image_path.stem}_{api_name}.json"
        return Path(self.config['cache_dir']) / cache_filename
        
    def _check_cache(self, image_path, api_name):
        """Check if results are cached for this image and API"""
        cache_path = self._get_cache_path(image_path, api_name)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
        
    def _save_to_cache(self, image_path, api_name, results):
        """Save API results to cache"""
        if not self.config['cache_results']:
            return
            
        cache_path = self._get_cache_path(image_path, api_name)
        if cache_path:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(results, f)
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")
                
    def _encode_image(self, image_path_or_array):
        """Encode image as base64 string for API requests"""
        if isinstance(image_path_or_array, (str, Path)):
            with open(image_path_or_array, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # Assume it's a numpy array
            success, buffer = cv2.imencode('.jpg', image_path_or_array)
            if not success:
                raise ValueError("Could not encode image")
            return base64.b64encode(buffer).decode('utf-8')
            
    def _process_with_openai(self, image_path_or_array):
        """Process image with OpenAI's GPT-4 Vision API"""
        if 'openai' not in self.api_keys:
            logger.warning("OpenAI API key not provided")
            return None
            
        # Check cache if it's a file path
        if isinstance(image_path_or_array, (str, Path)):
            cached_results = self._check_cache(image_path_or_array, 'openai')
            if cached_results:
                logger.info(f"Using cached OpenAI results for {image_path_or_array}")
                return cached_results
                
        try:
            # Encode image
            base64_image = self._encode_image(image_path_or_array)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys['openai']}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and provide the following information in JSON format:\n"
                                        "1. A list of all visible people and their approximate age ranges\n"
                                        "2. The main activity taking place\n"
                                        "3. The setting/environment\n"
                                        "4. Any notable objects\n"
                                        "5. The overall scene description\n"
                                        "Format your response as valid JSON with these keys: 'people', 'activity', 'setting', 'objects', 'description'"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 800
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
            response_data = response.json()
            
            # Extract the JSON from the response
            content = response_data['choices'][0]['message']['content']
            
            # Find JSON in the response
            try:
                # Try to parse the entire response as JSON
                result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from OpenAI response")
                        return None
                else:
                    logger.error("No JSON found in OpenAI response")
                    return None
            
            # Add metadata
            result['api_source'] = 'openai'
            result['model'] = response_data['model']
            result['timestamp'] = time.time()
            
            # Cache results if it's a file path
            if isinstance(image_path_or_array, (str, Path)):
                self._save_to_cache(image_path_or_array, 'openai', result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing with OpenAI: {e}")
            return None
            
    def _process_with_google(self, image_path_or_array):
        """Process image with Google Cloud Vision API"""
        if 'google' not in self.api_keys:
            logger.warning("Google API key not provided")
            return None
            
        # Check cache if it's a file path
        if isinstance(image_path_or_array, (str, Path)):
            cached_results = self._check_cache(image_path_or_array, 'google')
            if cached_results:
                logger.info(f"Using cached Google results for {image_path_or_array}")
                return cached_results
                
        try:
            # Encode image
            base64_image = self._encode_image(image_path_or_array)
            
            api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_keys['google']}"
            
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": base64_image
                        },
                        "features": [
                            {"type": "LABEL_DETECTION", "maxResults": 20},
                            {"type": "OBJECT_LOCALIZATION", "maxResults": 20},
                            {"type": "FACE_DETECTION", "maxResults": 20},
                            {"type": "TEXT_DETECTION", "maxResults": 20},
                            {"type": "SAFE_SEARCH_DETECTION"}
                        ]
                    }
                ]
            }
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Google API error: {response.status_code} - {response.text}")
                return None
                
            api_response = response.json()
            
            # Process the response into a more usable format
            result = {
                'api_source': 'google',
                'timestamp': time.time(),
                'labels': [],
                'objects': [],
                'faces': [],
                'text': '',
                'safe_search': {}
            }
            
            # Extract annotations
            annotations = api_response['responses'][0]
            
            # Labels
            if 'labelAnnotations' in annotations:
                for label in annotations['labelAnnotations']:
                    if label['score'] >= self.config['confidence_threshold']:
                        result['labels'].append({
                            'description': label['description'],
                            'confidence': label['score']
                        })
            
            # Objects
            if 'localizedObjectAnnotations' in annotations:
                for obj in annotations['localizedObjectAnnotations']:
                    if obj['score'] >= self.config['confidence_threshold']:
                        result['objects'].append({
                            'name': obj['name'],
                            'confidence': obj['score'],
                            'bounding_box': obj['boundingPoly']['normalizedVertices']
                        })
            
            # Faces
            if 'faceAnnotations' in annotations:
                for face in annotations['faceAnnotations']:
                    face_data = {
                        'bounding_box': face['boundingPoly']['vertices'],
                        'joy': face.get('joyLikelihood', 'UNKNOWN'),
                        'sorrow': face.get('sorrowLikelihood', 'UNKNOWN'),
                        'anger': face.get('angerLikelihood', 'UNKNOWN'),
                        'surprise': face.get('surpriseLikelihood', 'UNKNOWN')
                    }
                    result['faces'].append(face_data)
            
            # Text
            if 'fullTextAnnotation' in annotations:
                result['text'] = annotations['fullTextAnnotation']['text']
            
            # Safe search
            if 'safeSearchAnnotation' in annotations:
                result['safe_search'] = annotations['safeSearchAnnotation']
            
            # Generate a description based on the labels and objects
            top_labels = [label['description'] for label in result['labels'][:5]]
            top_objects = [obj['name'] for obj in result['objects'][:5]]
            
            all_entities = list(set(top_labels + top_objects))
            result['description'] = f"Image contains: {', '.join(all_entities)}"
            
            # Determine activity based on labels
            activity_labels = [label['description'] for label in result['labels'] 
                              if label['description'].lower().endswith('ing')]
            
            if activity_labels:
                result['activity'] = activity_labels[0]
            else:
                # Try to infer activity from top labels
                result['activity'] = "Unknown activity"
                
            # Determine setting based on labels
            setting_keywords = ['indoor', 'outdoor', 'room', 'building', 'nature', 
                               'office', 'home', 'street', 'park', 'beach']
            
            setting_labels = [label['description'] for label in result['labels'] 
                             if any(keyword in label['description'].lower() for keyword in setting_keywords)]
            
            if setting_labels:
                result['setting'] = setting_labels[0]
            else:
                result['setting'] = "Unknown setting"
                
            # People count based on faces
            result['people'] = []
            for i, _ in enumerate(result['faces']):
                result['people'].append({
                    'id': f"person_{i+1}",
                    'age_range': "unknown"
                })
                
            # Cache results if it's a file path
            if isinstance(image_path_or_array, (str, Path)):
                self._save_to_cache(image_path_or_array, 'google', result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing with Google: {e}")
            return None
            
    def _process_with_local_model(self, image_path_or_array):
        """Process image with local model (YOLOv5)"""
        if not self.local_model:
            logger.warning("Local model not available")
            return None
            
        # Check cache if it's a file path
        if isinstance(image_path_or_array, (str, Path)):
            cached_results = self._check_cache(image_path_or_array, 'local')
            if cached_results:
                logger.info(f"Using cached local model results for {image_path_or_array}")
                return cached_results
                
        try:
            # Load image
            if isinstance(image_path_or_array, (str, Path)):
                img = image_path_or_array
            else:
                # Save numpy array temporarily
                temp_path = Path('temp_image.jpg')
                cv2.imwrite(str(temp_path), image_path_or_array)
                img = temp_path
                
            # Run inference
            results = self.local_model(img)
            
            # Process results
            result = {
                'api_source': 'local',
                'timestamp': time.time(),
                'objects': [],
                'labels': [],
                'description': '',
                'activity': 'Unknown activity',
                'setting': 'Unknown setting',
                'people': []
            }
            
            # Extract detections
            detections = results.pandas().xyxy[0]
            
            # Count people
            person_count = 0
            
            for _, detection in detections.iterrows():
                if detection['confidence'] >= self.config['confidence_threshold']:
                    obj_data = {
                        'name': detection['name'],
                        'confidence': float(detection['confidence']),
                        'bounding_box': [
                            float(detection['xmin']),
                            float(detection['ymin']),
                            float(detection['xmax']),
                            float(detection['ymax'])
                        ]
                    }
                    result['objects'].append(obj_data)
                    
                    # Add to labels if not already present
                    label_exists = any(label['description'] == detection['name'] for label in result['labels'])
                    if not label_exists:
                        result['labels'].append({
                            'description': detection['name'],
                            'confidence': float(detection['confidence'])
                        })
                        
                    # Count people
                    if detection['name'].lower() == 'person':
                        person_count += 1
                        result['people'].append({
                            'id': f"person_{person_count}",
                            'age_range': "unknown"
                        })
            
            # Generate simple description
            if result['labels']:
                label_names = [label['description'] for label in result['labels']]
                result['description'] = f"Image contains: {', '.join(label_names)}"
            else:
                result['description'] = "No objects detected"
                
            # Clean up temp file if needed
            if isinstance(image_path_or_array, np.ndarray) and temp_path.exists():
                temp_path.unlink()
                
            # Cache results if it's a file path
            if isinstance(image_path_or_array, (str, Path)):
                self._save_to_cache(image_path_or_array, 'local', result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing with local model: {e}")
            
            # Clean up temp file if needed
            if 'temp_path' in locals() and isinstance(image_path_or_array, np.ndarray) and temp_path.exists():
                temp_path.unlink()
                
            return None
            
    def process_image(self, image_path_or_array):
        """Process an image using the preferred API order"""
        for api_name in self.config['api_preference']:
            if api_name == 'openai' and 'openai' in self.api_keys:
                result = self._process_with_openai(image_path_or_array)
                if result:
                    return result
            elif api_name == 'google' and 'google' in self.api_keys:
                result = self._process_with_google(image_path_or_array)
                if result:
                    return result
            elif api_name == 'local' and self.local_model:
                result = self._process_with_local_model(image_path_or_array)
                if result:
                    return result
                    
        logger.warning("All API attempts failed")
        return None
        
    def process_video_frames(self, frames_info, video_path=None, frames_dir=None):
        """Process multiple frames from a video
        
        Args:
            frames_info: List of frame info dictionaries (from IntelligentFrameSelector)
            video_path: Path to the source video (optional)
            frames_dir: Directory containing extracted frames (optional)
            
        Returns:
            List of dictionaries with frame info and API results
        """
        results = []
        
        for i, frame_info in enumerate(frames_info):
            logger.info(f"Processing frame {i+1}/{len(frames_info)}: {frame_info.get('frame_index', i)}")
            
            # Load the frame
            frame = None
            frame_path = None
            
            if 'filename' in frame_info and frames_dir:
                # Frame was saved as an image
                frame_path = Path(frames_dir) / frame_info['filename']
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
            elif video_path and 'frame_index' in frame_info:
                # Extract frame from video
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info['frame_index'] - 1)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    logger.error(f"Failed to extract frame {frame_info['frame_index']} from video")
                    continue
                    
                # Save frame for caching
                if frames_dir:
                    frames_dir = Path(frames_dir)
                    frames_dir.mkdir(exist_ok=True, parents=True)
                    frame_path = frames_dir / f"{Path(video_path).stem}_frame{frame_info['frame_index']:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_info['filename'] = frame_path.name
            
            if frame is None and frame_path is None:
                logger.error("No frame data available to process")
                continue
                
            # Process the frame
            api_result = None
            if frame_path:
                api_result = self.process_image(frame_path)
            elif frame is not None:
                api_result = self.process_image(frame)
                
            if api_result:
                # Combine frame info with API results
                combined_result = {**frame_info, 'vision_api': api_result}
                results.append(combined_result)
            else:
                logger.warning(f"Failed to process frame {frame_info.get('frame_index', i)}")
                
        return results
        
    def summarize_video_content(self, frame_results):
        """Generate a summary of video content based on processed frames
        
        Args:
            frame_results: List of frame results from process_video_frames
            
        Returns:
            Dictionary with video content summary
        """
        if not frame_results:
            return {"error": "No frame results to summarize"}
            
        # Collect all people detected
        all_people = []
        for result in frame_results:
            if 'vision_api' in result and 'people' in result['vision_api']:
                all_people.extend(result['vision_api']['people'])
                
        # Count unique people (very basic approach - could be improved)
        unique_people = {}
        for person in all_people:
            person_id = person.get('id', 'unknown')
            if person_id not in unique_people:
                unique_people[person_id] = person
                
        # Collect all activities
        activities = []
        for result in frame_results:
            if 'vision_api' in result and 'activity' in result['vision_api']:
                activity = result['vision_api']['activity']
                if activity and activity != "Unknown activity":
                    activities.append(activity)
                    
        # Count activity occurrences
        activity_counts = {}
        for activity in activities:
            if activity not in activity_counts:
                activity_counts[activity] = 0
            activity_counts[activity] += 1
            
        # Find most common activity
        main_activity = "Unknown activity"
        if activity_counts:
            main_activity = max(activity_counts.items(), key=lambda x: x[1])[0]
            
        # Collect all settings
        settings = []
        for result in frame_results:
            if 'vision_api' in result and 'setting' in result['vision_api']:
                setting = result['vision_api']['setting']
                if setting and setting != "Unknown setting":
                    settings.append(setting)
                    
        # Count setting occurrences
        setting_counts = {}
        for setting in settings:
            if setting not in setting_counts:
                setting_counts[setting] = 0
            setting_counts[setting] += 1
            
        # Find most common setting
        main_setting = "Unknown setting"
        if setting_counts:
            main_setting = max(setting_counts.items(), key=lambda x: x[1])[0]
            
        # Collect all objects
        all_objects = []
        for result in frame_results:
            if 'vision_api' in result and 'objects' in result['vision_api']:
                for obj in result['vision_api']['objects']:
                    if isinstance(obj, dict) and 'name' in obj:
                        all_objects.append(obj['name'])
                    elif isinstance(obj, str):
                        all_objects.append(obj)
                        
        # Count object occurrences
        object_counts = {}
        for obj in all_objects:
            if obj not in object_counts:
                object_counts[obj] = 0
            object_counts[obj] += 1
            
        # Find top objects
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Generate summary
        summary = {
            "people_count": len(unique_people),
            "main_activity": main_activity,
            "setting": main_setting,
            "top_objects": [obj[0] for obj in top_objects],
            "frame_count": len(frame_results),
            "timestamp": time.time()
        }
        
        return summary

def main():
    """Run the processor on a sample image or video frames"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images with Vision APIs')
    parser.add_argument('--image', help='Path to an image file to process')
    parser.add_argument('--frames-dir', help='Directory containing video frames')
    parser.add_argument('--frames-json', help='JSON file with frame information')
    parser.add_argument('--output', help='Output file for results', default='vision_results.json')
    parser.add_argument('--api', choices=['openai', 'google', 'local'], help='Preferred API to use')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--google-key', help='Google API key')
    parser.add_argument('--local-model', help='Path to local model weights')
    args = parser.parse_args()
    
    # Set up API keys
    api_keys = {}
    if args.openai_key:
        api_keys['openai'] = args.openai_key
    if args.google_key:
        api_keys['google'] = args.google_key
        
    # Set up config
    config = {}
    if args.api:
        config['api_preference'] = [args.api]
    if args.local_model:
        config['local_model_path'] = args.local_model
        
    # Create processor
    processor = VisionAPIProcessor(api_keys, config)
    
    # Process image
    if args.image:
        result = processor.process_image(args.image)
        if result:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Image processing results saved to {args.output}")
        else:
            print("Failed to process image")
            
    # Process video frames
    elif args.frames_dir and args.frames_json:
        with open(args.frames_json, 'r') as f:
            frames_info = json.load(f)
            
        if 'selected_frames' in frames_info:
            frames_info = frames_info['selected_frames']
            
        results = processor.process_video_frames(frames_info, frames_dir=args.frames_dir)
        summary = processor.summarize_video_content(results)
        
        output = {
            "frame_results": results,
            "summary": summary
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Video frame processing results saved to {args.output}")
        
    else:
        print("Please provide either an image path or frames directory and JSON info")

if __name__ == "__main__":
    main() 