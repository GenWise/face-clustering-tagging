# Face Clustering & Tagging System

This project allows you to process group photos, extract faces, cluster similar faces, and tag them efficiently using both context-aware and cluster-aware interfaces.

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

---

## 📂 Directory Structure
```
project_root/
├── test_images/                      # Raw group photos
├── detected_faces/                  # Outputs: face crops, overlays, metadata
│   ├── *.jpg                        # Cropped faces
│   ├── debug_resized_*.JPG         # Resized group images
│   ├── face_metadata.json          # Metadata for each face
│   ├── face_embeddings.pkl         # Vector embeddings
│   ├── face_clusters.json          # Cluster mapping
├── encode_faces_insightface.py     # Main extraction script
├── cluster_embeddings.py           # Clustering logic
├── cleanup_and_recluster.py        # Prune and rebuild
├── cluster_tagging_ui.html         # UI for tagging clusters
├── face_tagging_ui.html            # UI for tagging by group image
├── face_tagging_server.py          # Flask app to serve UIs and metadata
```

---

## 🚀 Next Steps
- Add `git` support
- Push to GitHub repository
- Enable CSV exports and summaries

Let me know when you're ready to set up your GitHub repo and I'll walk you through the commit & push process!
