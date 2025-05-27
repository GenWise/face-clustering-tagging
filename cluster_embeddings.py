# cluster_embeddings.py
# Groups face embeddings into clusters and outputs face_clusters.json

import pickle
import json
from sklearn.cluster import DBSCAN
import numpy as np
from pathlib import Path

# Load embeddings
with open("detected_faces/face_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)  # dict: { face_id: embedding list }

face_ids = list(embeddings.keys())
embedding_matrix = np.array([embeddings[fid] for fid in face_ids])

# Cluster embeddings
print("Clustering embeddings with DBSCAN...")
dbscan = DBSCAN(eps=0.4, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(embedding_matrix)

# Map face_id to cluster
clusters = {face_id: int(label) for face_id, label in zip(face_ids, labels)}

# Save cluster assignments
output_path = Path("detected_faces/face_clusters.json")
with open(output_path, "w") as f:
    json.dump(clusters, f, indent=2)

print(f"Saved {len(set(labels)) - (1 if -1 in labels else 0)} clusters to {output_path}")
