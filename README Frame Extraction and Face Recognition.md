# 🎥 Frame Extraction & Face Recognition Pipeline

This repository implements a complete video analysis pipeline for:

* Extracting keyframes
* Detecting and recognizing faces
* Tagging known individuals
* Managing unmatched face crops for later tagging

## 🗂️ Key Files Created / Edited

* `kf_extract_and_processing.py`: Enhanced core pipeline (rewritten here)
* `run_benchmark.py`: Entry point to run batch processing
* `pipeline.log`: Log file tracking all steps and scores
* `video_tags.json`: Recognized face tags per frame
* `unique_tags_by_video.json`: Summary of unique tags per video

## ⚙️ Functionality Overview

1. **Keyframe Extraction**

   * Extracts frames every N frames from video clips in `videos/`.

2. **Face Detection + Blur Filter**

   * Uses YOLOv8n-face to detect faces
   * Filters out blurry or low-quality faces using Laplacian variance

3. **Face Recognition**

   * InsightFace (`buffalo_l`) with cosine similarity
   * Recognizes faces from a prebuilt `face_db` using embeddings + metadata

4. **Face Annotation**

   * Saves annotated frames with human-readable tags
   * Stores them in `annotated_keyframes/{video}/`

5. **Unmatched Face Handling**

   * Crops unrecognized faces to `detected_faces/{video}/`
   * Also copied to root `detected_faces/` for UI use

6. **Summarization**

   * `video_tags.json`: Tags detected per frame
   * `unique_tags_by_video.json`: Unique identities per video

## 💡 Key Design Insights

* Cosine similarity is superior for normalized face embeddings
* Laplacian filtering improves recognition accuracy by eliminating noise
* Linking unmatched faces into a shared folder enables reuse of existing tagging UIs

## 🔄 Tagging Pipeline Integration

After running `run_benchmark.py`, use your browser-based UI (e.g. `cluster_tagging_ui.html`) to label unmatched faces from `detected_faces/`.

## 📝 Version Control Recommendations

```bash
git add kf_extract_and_processing.py run_benchmark.py *.json
git commit -m "Enhanced frame extraction and face recognition pipeline"
git tag v1.0-keyframe-face-pipeline
```

---

This pipeline enables scalable face recognition in keyframes across multiple videos and integrates cleanly with existing manual tagging tools.
