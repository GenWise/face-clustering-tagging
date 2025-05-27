import os
import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis
from collections import defaultdict
import json
import pickle
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    filename="pipeline.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# Paths
VIDEO_DIR = "videos"
KEYFRAME_DIR = "keyframes"
ANNOTATED_DIR = "annotated_keyframes"
UNMATCHED_FACE_DIR_BASE = "detected_faces"
FACE_DB_PATH = "face_db"
VIDEO_TAGS_JSON = "video_tags.json"

# Parameters
FRAME_INTERVAL = 30
BLUR_THRESHOLD = 20.0
FACE_SIZE_THRESHOLD = 100
SIMILARITY_THRESHOLD = 0.6

# Initialize models
face_detector = YOLO("yolov8n-face.pt")
face_recognizer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_recognizer.prepare(ctx_id=0)

# Load face_db from pickle and metadata
with open(os.path.join(FACE_DB_PATH, "face_embeddings.pkl"), "rb") as f:
    embedding_records = pickle.load(f)

with open(os.path.join(FACE_DB_PATH, "face_metadata.json"), "r") as f:
    metadata = json.load(f)

# Build a face_id → tag lookup
metadata_lookup = {}
for entry in metadata:
    face_id = entry.get("face_id")
    tag = entry.get("tag", face_id)
    if face_id:
        metadata_lookup[face_id] = tag

face_db = {}
tag_lookup = {}  # For label use
if isinstance(embedding_records, dict):
    for name, embedding in embedding_records.items():
        tag = metadata_lookup.get(name, name)
        face_db.setdefault(tag, []).append(embedding)
        tag_lookup[name] = tag
else:
    for record in embedding_records:
        if not isinstance(record, dict):
            continue
        cluster_id = str(record.get("cluster_id", "unknown"))
        embedding = record.get("embedding")
        if embedding is None:
            continue
        face_id = record.get("face_id", cluster_id)
        tag = metadata_lookup.get(face_id, f"cluster_{cluster_id}")
        if tag not in face_db:
            face_db[tag] = []
        face_db[tag].append(embedding)
        tag_lookup[cluster_id] = tag

seen_faces = set()  # Prevent duplicate saves across frames

def is_blurry(image):
    score = cv2.Laplacian(image, cv2.CV_64F).var()
    logging.info(f"Laplacian score: {score:.2f}")
    return score < BLUR_THRESHOLD

def extract_keyframes(video_path, keyframe_output_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % FRAME_INTERVAL == 0:
            out_path = os.path.join(keyframe_output_path, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_id += 1
        count += 1
    cap.release()

def match_face(face_embedding):
    best_match = None
    best_score = -1.0
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    for tag, embeddings in face_db.items():
        for db_embedding in embeddings:
            db_embedding = db_embedding / np.linalg.norm(db_embedding)
            similarity = np.dot(face_embedding, db_embedding)
            if similarity > best_score:
                best_score = similarity
                best_match = tag
    logging.info(f"Best match: {best_match} with cosine similarity: {best_score:.4f}")
    return best_match if best_score > (1 - SIMILARITY_THRESHOLD) else None

def process_keyframes(video_file):
    video_id = os.path.splitext(video_file)[0]
    video_keyframes_dir = os.path.join(KEYFRAME_DIR, video_id)
    os.makedirs(video_keyframes_dir, exist_ok=True)
    extract_keyframes(os.path.join(VIDEO_DIR, video_file), video_keyframes_dir)

    annotated_output_dir = os.path.join(ANNOTATED_DIR, video_id)
    os.makedirs(annotated_output_dir, exist_ok=True)

    unmatched_face_dir = os.path.join(UNMATCHED_FACE_DIR_BASE, video_id)
    os.makedirs(unmatched_face_dir, exist_ok=True)

    video_summary = defaultdict(list)

    for frame_name in sorted(os.listdir(video_keyframes_dir)):
        frame_path = os.path.join(video_keyframes_dir, frame_name)
        frame = cv2.imread(frame_path)

        if is_blurry(frame):
            logging.info(f"Skipping blurry frame: {frame_name}")
            continue

        results = face_detector(frame)
        if not results or not results[0].boxes:
            logging.info(f"No YOLO faces in frame: {frame_name}")
            continue

        faces = face_recognizer.get(frame)
        if not faces:
            logging.info(f"InsightFace found no faces in: {frame_name}")
            continue

        any_saved = False
        for i, face in enumerate(faces):
            if face.bbox[2] - face.bbox[0] < FACE_SIZE_THRESHOLD:
                logging.info(f"Face too small in {frame_name}, skipping.")
                continue

            tag = match_face(face.embedding)
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            if tag:
                logging.info(f"Recognized {tag} in {frame_name}")
                video_summary[frame_name].append(tag)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                any_saved = True
            else:
                logging.info(f"Unmatched face in {frame_name}, saving crop")
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logging.warning(f"Empty crop for face {i} in {frame_name}, skipping save")
                    continue
                hash_key = f"{crop.shape}_{crop.mean():.1f}_{crop.std():.1f}"
                if hash_key in seen_faces:
                    logging.info(f"Duplicate face skipped in {frame_name}")
                    continue
                seen_faces.add(hash_key)
                unmatched_path = os.path.join(unmatched_face_dir, f"{video_id}_{frame_name}_face{i}.jpg")
                if not os.path.exists(unmatched_path):
                    cv2.imwrite(unmatched_path, crop)

        if any_saved:
            annotated_frame_path = os.path.join(annotated_output_dir, frame_name)
            cv2.imwrite(annotated_frame_path, frame)

    return video_summary

def main():
    all_tags = {}
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.lower().endswith(('.mp4', '.mov', '.avi')):
            continue
        video_id = os.path.splitext(video_file)[0]
        logging.info(f"Processing video: {video_file}")
        summary = process_keyframes(video_file)
        all_tags[video_id] = summary

    with open(VIDEO_TAGS_JSON, "w") as f:
        json.dump(all_tags, f, indent=2)

    # Write unique tags per video_id
    unique_tags_per_video = {
        vid: sorted(set(tag for tags in frame_tags.values() for tag in tags))
        for vid, frame_tags in all_tags.items()
    }
    with open("unique_tags_by_video.json", "w") as f:
        json.dump(unique_tags_per_video, f, indent=2)

if __name__ == "__main__":
    main()
