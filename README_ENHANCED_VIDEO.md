# 🎥 Enhanced Video Processing Pipeline

This module provides advanced video processing capabilities with intelligent keyframe selection, face recognition, and content analysis using Vision APIs.

## 🚀 Features

- **Intelligent Keyframe Selection**
  - Scene change detection
  - Motion intensity analysis
  - Face count changes
  - Visual diversity metrics

- **Face Recognition**
  - Detect and recognize faces in keyframes
  - Match against existing face database
  - Filter out blurry faces
  - Save unrecognized faces for later tagging

- **Content Analysis**
  - Analyze activities in the video
  - Identify settings/environments
  - Detect objects and people
  - Generate comprehensive scene descriptions

- **Flexible API Options**
  - OpenAI GPT-4 Vision
  - Google Cloud Vision API
  - Local model support (YOLOv5)
  - Configurable API preference order

## 📋 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Basic Usage

Process a single video:

```bash
python enhanced_video_processor.py videos/my_video.mp4
```

### Advanced Options

```bash
python enhanced_video_processor.py videos/my_video.mp4 \
  --output-dir processed_videos \
  --keyframes-dir keyframes \
  --annotated-dir annotated_keyframes \
  --detected-faces-dir detected_faces \
  --face-db-dir face_db \
  --max-api-calls 60 \
  --face-threshold 0.6 \
  --save-all-frames \
  --api openai \
  --openai-key YOUR_OPENAI_API_KEY
```

### Batch Processing

To process multiple videos:

```bash
python batch_video_processor.py --input videos/ --workers 4
```

## 🔍 How It Works

1. **Keyframe Selection**
   - Analyzes the video to select the most informative frames
   - Uses scene changes, motion, and face detection to make smart selections
   - Limits API calls to stay within budget

2. **Vision API Processing**
   - Sends selected keyframes to Vision APIs
   - Analyzes content, activities, and settings
   - Caches results to avoid duplicate API calls

3. **Face Processing**
   - Detects faces in each keyframe
   - Matches against existing face database
   - Saves unrecognized faces for later tagging

4. **Result Compilation**
   - Combines face recognition and vision analysis results
   - Updates video tags and unique tags databases
   - Generates comprehensive summary

## 📁 Output Structure

```
project_root/
├── keyframes/                      # Extracted video frames
│   ├── {video_name}/               # Frames from specific video
├── annotated_keyframes/            # Frames with face annotations
│   ├── {video_name}/               # Annotated frames from specific video
├── detected_faces/                 # Unrecognized face crops
│   ├── {video_name}/               # Faces from specific video
├── processed_videos/               # Processing results
│   ├── {video_name}_results.json   # Complete analysis results
│   ├── vision_api_cache/           # Cached API responses
├── video_tags.json                 # Tags per frame for all videos
├── unique_tags_by_video.json       # Unique tags per video
```

## 🛠️ Configuration

You can configure the system by modifying the parameters in `enhanced_video_processor.py` or by passing command-line arguments.

Key configuration options:

- `max_api_calls_per_video`: Maximum Vision API calls per video
- `face_recognition_threshold`: Confidence threshold for face recognition
- `face_blur_threshold`: Threshold for detecting blurry faces
- `api_preference`: Order of API preference (openai, google, local)

## 🧠 Extending the System

### Adding New Vision APIs

1. Add a new method in `VisionAPIProcessor` (e.g., `_process_with_azure`)
2. Update the `process_image` method to include the new API
3. Add the new API to the `api_preference` configuration

### Improving Face Recognition

1. Fine-tune the face recognition threshold
2. Implement face clustering for unrecognized faces
3. Add age and gender estimation

### Enhancing Content Analysis

1. Add custom activity classifiers
2. Implement temporal analysis across frames
3. Add audio transcription and analysis

## 📊 Performance Considerations

- Processing time depends on video length and resolution
- API calls are the main bottleneck and cost factor
- Adjust `max_api_calls_per_video` based on your budget
- Use local models when possible for faster processing

## 🔒 API Keys

For security, it's recommended to set API keys as environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_google_key
```

Or pass them directly via command-line arguments.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 