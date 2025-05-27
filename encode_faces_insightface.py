# encode_faces_insightface.py
# Detect, encode, save, and prepare for tagging faces from group photos using InsightFace (ONNX backend)

import sys
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from itertools import combinations
from insightface.app import FaceAnalysis

img_folder = sys.argv[1] if len(sys.argv) > 1 else "test_images/"
img_paths = sorted(list(Path(img_folder).glob("*.jpg")) + list(Path(img_folder).glob("*.JPG")))

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

face_output_dir = Path("detected_faces")
metadata_dir = Path("face_db")
face_output_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

face_metadata = []
embeddings = {}

print("\n[InsightFace] Detecting and encoding faces from group photos:")
for img_path in img_paths:
    print(f"🔎 Processing: {img_path.name}")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not load image: {img_path.name}")
        continue

    print(f"⚠️ Loaded image: {img.shape} | dtype: {img.dtype}")

    h, w = img.shape[:2]
    if w > 1080:
        scale = 1080 / w
        img = cv2.resize(img, (1080, int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"🔍 Resized {img_path.name} to width 1080 for better detection")

    debug_resized_path = face_output_dir / f"debug_resized_{img_path.name}"
    cv2.imwrite(str(debug_resized_path), img)

    brightness = np.mean(img)
    print(f"🧪 {img_path.name} - shape: {img.shape}, mean brightness: {brightness:.1f}")

    try:
        faces = app.get(img)
        print(f"📸 InsightFace detected {len(faces)} face(s) in {img_path.name}")
    except Exception as e:
        print(f"❌ InsightFace failed on {img_path.name}: {e}")
        continue

    if not faces:
        continue

    height, width = img.shape[:2]

    for i, face in enumerate(faces):
        face_id = f"{img_path.stem}_face{i}"
        embedding_vector = face.embedding.tolist()
        embeddings[face_id] = embedding_vector
        print(f"✅ {face_id} encoded. Shape: {face.embedding.shape}")

        x1 = max(0, min(int(face.bbox[0]), width - 1))
        y1 = max(0, min(int(face.bbox[1]), height - 1))
        x2 = max(0, min(int(face.bbox[2]), width - 1))
        y2 = max(0, min(int(face.bbox[3]), height - 1))

        if x2 > x1 and y2 > y1:
            face_crop = img[y1:y2, x1:x2]
            face_crop_path = face_output_dir / f"{face_id}.jpg"
            cv2.imwrite(str(face_crop_path), face_crop)
        else:
            print(f"⚠️ Skipped {face_id} due to invalid bounding box.")
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, face_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Try auto-tagging if similar to a known face
        auto_tag = None
        for known in face_metadata:
            if known['tag'] and known['embedding'] in embeddings:
                vec_a = np.array(embedding_vector)
                vec_b = np.array(embeddings[known['embedding']])
                cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                distance = 1 - cosine_sim
                if distance < 0.35:
                    auto_tag = known['tag']
                    print(f"🔁 Auto-tagged {face_id} as '{auto_tag}' (distance: {distance:.4f})")
                    break

        face_metadata.append({
            "face_id": face_id,
            "source_image": img_path.name,
            "face_path": str(face_crop_path),
            "bbox": [x1, y1, x2, y2],
            "embedding": face_id,
            "tag": auto_tag
        })

    vis_path = face_output_dir / f"{img_path.stem}_vis.jpg"
    cv2.imwrite(str(vis_path), img)

with open(metadata_dir / "face_metadata.json", "w") as f:
    json.dump(face_metadata, f, indent=2)

with open(metadata_dir / "face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("\n[InsightFace] Pairwise cosine distances:")
for a, b in combinations(embeddings.keys(), 2):
    vec_a = np.array(embeddings[a])
    vec_b = np.array(embeddings[b])
    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    distance = 1 - cosine_sim
    print(f"{a} <-> {b}: Cosine Distance = {distance:.4f}")
