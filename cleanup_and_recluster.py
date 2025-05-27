# cleanup_and_recluster.py
# Removes metadata entries for missing files and regenerates clusters

import json
import pickle
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

faces_dir = Path("detected_faces")
meta_file = faces_dir / "face_metadata.json"
emb_file = faces_dir / "face_embeddings.pkl"
clust_file = faces_dir / "face_clusters.json"

# Load metadata and embeddings
with open(meta_file, "r") as f:
    metadata = json.load(f)
with open(emb_file, "rb") as f:
    embeddings = pickle.load(f)

# Clean metadata and embeddings if face image missing or empty tag
cleaned_metadata = []
cleaned_embeddings = {}

skipped_unknown = 0
skipped_blank = 0
removed_missing = 0

for item in metadata:
    path = Path(item["face_path"])
    tag = str(item.get("tag", "") or "").strip()


    if not path.exists():
        removed_missing += 1
        continue
    if tag == "":
        skipped_blank += 1
        continue
    if tag.lower() == "unknown":
        skipped_unknown += 1
        continue

    cleaned_metadata.append(item)
    if item["embedding"] in embeddings:
        cleaned_embeddings[item["embedding"]] = embeddings[item["embedding"]]

# Save cleaned metadata and embeddings
with open(meta_file, "w") as f:
    json.dump(cleaned_metadata, f, indent=2)
with open(emb_file, "wb") as f:
    pickle.dump(cleaned_embeddings, f)

# Re-cluster
print("\n🔄 Re-clustering...")
face_ids = list(cleaned_embeddings.keys())
embedding_matrix = np.array([cleaned_embeddings[fid] for fid in face_ids])
dbscan = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(embedding_matrix)

face_clusters = {fid: int(label) for fid, label in zip(face_ids, labels)}
with open(clust_file, "w") as f:
    json.dump(face_clusters, f, indent=2)

print(f"✅ Cleaned metadata: {len(cleaned_metadata)} entries")
print(f"✅ Removed: {removed_missing} missing, {skipped_blank} blank, {skipped_unknown} 'Unknown'")
print(f"✅ Saved {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
