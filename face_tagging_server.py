# face_tagging_server.py
# Flask server to serve face tagging UI and update metadata

from flask import Flask, send_from_directory, request, jsonify
from pathlib import Path
import json

app = Flask(__name__, static_url_path='', static_folder='.')
metadata_path = Path("detected_faces/face_metadata.json")

@app.route('/')
def index():
    return send_from_directory('.', 'face_tagging_ui.html')

@app.route('/detected_faces/<path:filename>')
def face_images(filename):
    return send_from_directory('detected_faces', filename)

@app.route('/cluster_tagging_ui.html')
def cluster_ui():
    return send_from_directory('.', 'cluster_tagging_ui.html')

@app.route('/delete-untagged', methods=['POST'])
def delete_untagged():
    from os import remove
    from pathlib import Path
    files = request.json.get('files', [])
    deleted = []
    for rel_path in files:
        try:
            f = Path(rel_path)
            if f.exists(): f.unlink()
            deleted.append(str(f))
        except Exception as e:
            print(f"Failed to delete {rel_path}: {e}")
    return jsonify({"deleted": deleted})


@app.route('/save-tags', methods=['POST'])
def save_tags():
    if not metadata_path.exists():
        return jsonify({"error": "face_metadata.json not found"}), 404

    updated_tags = {item['face_id']: item['tag'] for item in request.json}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    for item in metadata:
        face_id = item['face_id']
        if face_id in updated_tags:
            item['tag'] = updated_tags[face_id]

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
