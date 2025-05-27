import os
import time
import subprocess

VIDEO_DIR = "videos"
LOG_FILE = "video_runtime_log.txt"

def main():
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    with open(LOG_FILE, "w") as log:
        for video in sorted(videos):
            print(f"▶️  Processing: {video}")
            start = time.time()
            subprocess.run(["python", "kf_extract_and_processing.py"], check=True)
            end = time.time()
            duration = end - start
            log.write(f"{video}: {duration:.2f} seconds\n")
            print(f"✅  Done: {video} in {duration:.2f} seconds\n")

if __name__ == "__main__":
    main()
