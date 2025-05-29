# Face Clustering & Tagging System

This project allows you to process group photos and videos, extract faces, cluster similar faces, and tag them efficiently using both context-aware and cluster-aware interfaces.

---

## 🔧 Components

### 1. **Face Extraction & Embedding**
Run this to:
- Detect all faces in `test_images/`
- Save cropped faces and overlays to `detected_faces/`
- Generate metadata and vector embeddings

```bash
python encode_faces_insightface.py test_images/
```

### 2. **Clustering by Similarity**
Groups faces across all images based on visual similarity:

```bash
python cluster_embeddings.py
```

### 3. **Cluster Tagging UI**
Visit:
```text
http://localhost:5001/cluster_tagging_ui.html
```

- Tag all faces in a cluster at once
- Mark as `Unknown` if unsure
- "Retag Unknowns" button helps revisit later
- Save and clean UI options available

### 4. **Contextual Image Tagging UI**
Visit:
```text
http://localhost:5001/
```

- View and tag each face in the context of the original group photo
- Auto-suggested tags from clusters and previous tags
- Keyboard shortcuts, filtering, and suggestions built-in

### 5. **Clean-up & Reclustering**
To remove deleted or irrelevant entries and regenerate clusters:
```bash
python cleanup_and_recluster.py
```

### 6. **Video Processing Pipeline**
Process videos to extract keyframes and detect faces:
```bash
python kf_extract_and_processing.py videos/my_video.mp4
```

### 7. **Batch Video Processing**
Process multiple videos in parallel:
```bash
python batch_video_processor.py --input videos/ --workers 4
```

### 8. **Unified Search Interface**
Search across both images and videos:
```bash
python unified_search.py --tags person1 person2 --type video
```

### 9. **Tag Statistics Dashboard**
Generate visualizations of tag statistics:
```bash
python tag_statistics_dashboard.py --output statistics/
```

### 10. **Export Reports**
Export tag data in various formats:
```bash
python export_reports.py --format all --output exports/
```

---

## 📂 Directory Structure
```
project_root/
├── test_images/                      # Raw group photos
├── videos/                           # Input video files
├── keyframes/                        # Extracted video frames
│   ├── {video_name}/                 # Frames from specific video
├── annotated_keyframes/              # Frames with face annotations
│   ├── {video_name}/                 # Annotated frames from specific video
├── detected_faces/                   # Outputs: face crops, overlays, metadata
│   ├── *.jpg                         # Cropped faces
│   ├── debug_resized_*.JPG           # Resized group images
│   ├── face_metadata.json            # Metadata for each face
│   ├── face_embeddings.pkl           # Vector embeddings
│   ├── face_clusters.json            # Cluster mapping
├── statistics/                       # Tag statistics and visualizations
├── exports/                          # Exported reports and data
├── encode_faces_insightface.py       # Main extraction script
├── cluster_embeddings.py             # Clustering logic
├── cleanup_and_recluster.py          # Prune and rebuild
├── kf_extract_and_processing.py      # Video keyframe extraction
├── batch_video_processor.py          # Batch video processing
├── unified_search.py                 # Search across images and videos
├── tag_statistics_dashboard.py       # Generate tag statistics
├── export_reports.py                 # Export data in various formats
├── cluster_tagging_ui.html           # UI for tagging clusters
├── face_tagging_ui.html              # UI for tagging by group image
├── face_tagging_server.py            # Flask app to serve UIs and metadata
```

---

## 🚀 Next Steps
- Implement face recognition model fine-tuning
- Add automatic activity detection in videos
- Integrate with cloud storage
- Create a web-based dashboard for all functionality

---

## 📝 Version Management

To commit changes to the repository:

```bash
# Add all new files
git add unified_search.py batch_video_processor.py tag_statistics_dashboard.py export_reports.py

# Commit with descriptive message
git commit -m "Added unified search, batch processing, statistics dashboard, and export functionality"

# Create a version tag
git tag v1.1-enhanced-tools

# Push to remote repository
git push origin main
git push origin v1.1-enhanced-tools
```
