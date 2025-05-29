#!/usr/bin/env python3
# batch_video_processor.py - Process multiple videos in batch mode

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchVideoProcessor:
    """Process multiple videos in parallel or sequence"""
    
    def __init__(self, input_dir="videos", max_workers=2, skip_existing=True):
        self.input_dir = Path(input_dir)
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        self.supported_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        self.start_time = time.time()
        self.processed_videos = 0
        self.failed_videos = []
        
    def find_videos(self):
        """Find all video files in input directory"""
        videos = []
        
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return videos
            
        for ext in self.supported_extensions:
            videos.extend(list(self.input_dir.glob(f"**/*{ext}")))
            
        logger.info(f"Found {len(videos)} videos to process")
        return videos
    
    def should_process(self, video_path):
        """Check if video should be processed based on skip_existing flag"""
        if not self.skip_existing:
            return True
            
        # Check if this video has already been processed
        video_name = video_path.stem
        keyframes_dir = Path("keyframes") / video_name
        
        # If keyframes directory exists and has files, consider it processed
        if keyframes_dir.exists() and any(keyframes_dir.iterdir()):
            logger.info(f"Skipping {video_name} - already processed")
            return False
            
        return True
    
    def process_video(self, video_path):
        """Process a single video using the keyframe extraction script"""
        video_name = video_path.stem
        logger.info(f"Processing {video_name}")
        
        try:
            # Call the main processing script
            cmd = [sys.executable, "kf_extract_and_processing.py", str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error processing {video_name}: {result.stderr}")
                self.failed_videos.append(str(video_path))
                return False
            
            logger.info(f"Successfully processed {video_name}")
            self.processed_videos += 1
            return True
            
        except Exception as e:
            logger.error(f"Exception processing {video_name}: {e}")
            self.failed_videos.append(str(video_path))
            return False
    
    def process_all(self):
        """Process all videos using thread pool"""
        videos = self.find_videos()
        videos_to_process = [v for v in videos if self.should_process(v)]
        
        if not videos_to_process:
            logger.info("No videos to process")
            return
            
        logger.info(f"Processing {len(videos_to_process)} videos with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_video, video): video for video in videos_to_process}
            
            for future in as_completed(futures):
                video = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing {video.stem}: {e}")
                    self.failed_videos.append(str(video))
        
        # Generate summary report
        self.generate_report()
    
    def generate_report(self):
        """Generate processing summary report"""
        elapsed_time = time.time() - self.start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_videos_found": len(self.find_videos()),
            "videos_processed": self.processed_videos,
            "videos_failed": len(self.failed_videos),
            "failed_videos": self.failed_videos,
            "processing_time_seconds": elapsed_time
        }
        
        # Save report to JSON
        with open("batch_processing_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {self.processed_videos} videos")
        logger.info(f"Failed {len(self.failed_videos)} videos")
        
        if self.failed_videos:
            logger.info("Failed videos:")
            for video in self.failed_videos:
                logger.info(f"  - {video}")

def main():
    parser = argparse.ArgumentParser(description='Batch process multiple videos')
    parser.add_argument('--input', default='videos', help='Input directory containing videos')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--process-all', action='store_true', help='Process all videos, including already processed ones')
    args = parser.parse_args()
    
    processor = BatchVideoProcessor(
        input_dir=args.input,
        max_workers=args.workers,
        skip_existing=not args.process_all
    )
    
    processor.process_all()

if __name__ == "__main__":
    main() 