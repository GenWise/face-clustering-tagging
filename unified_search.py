#!/usr/bin/env python3
# unified_search.py - Search interface for both images and videos

import json
import os
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional

class UnifiedMediaSearch:
    """Search across both images and videos using tags and metadata"""
    
    def __init__(self):
        self.video_tags_path = "video_tags.json"
        self.unique_video_tags_path = "unique_tags_by_video.json"
        self.face_metadata_path = "face_db/face_metadata.json"
        
        # Load data sources
        self.video_tags = self._load_json(self.video_tags_path, {})
        self.unique_video_tags = self._load_json(self.unique_video_tags_path, {})
        self.face_metadata = self._load_json(self.face_metadata_path, {})
        
        # Create pandas DataFrame for easier searching
        self._build_search_index()
    
    def _load_json(self, path: str, default: Any = None) -> Any:
        """Load JSON file with error handling"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return default
    
    def _build_search_index(self):
        """Build unified search index from all data sources"""
        # Process video data
        video_records = []
        
        # Add video records
        for video_id, tags in self.unique_video_tags.items():
            video_records.append({
                'media_type': 'video',
                'media_id': video_id,
                'tags': tags,
                'source_path': f"videos/{video_id}",
                'keyframes_path': f"keyframes/{video_id}"
            })
        
        # Process image data
        image_records = []
        
        # Group face metadata by source image
        image_to_faces = {}
        for face_id, metadata in self.face_metadata.items():
            if 'source_image' in metadata:
                source = metadata['source_image']
                if source not in image_to_faces:
                    image_to_faces[source] = []
                
                # Add face tags if available
                tags = metadata.get('tag', 'unknown')
                if tags not in image_to_faces[source]:
                    image_to_faces[source].append(tags)
        
        # Create image records
        for image_path, tags in image_to_faces.items():
            image_records.append({
                'media_type': 'image',
                'media_id': os.path.basename(image_path),
                'tags': tags,
                'source_path': image_path
            })
        
        # Combine into unified index
        self.search_index = pd.DataFrame(video_records + image_records)
    
    def search_by_tag(self, tag: str) -> pd.DataFrame:
        """Search for media containing specific tag"""
        results = self.search_index[self.search_index['tags'].apply(
            lambda tags: tag.lower() in [t.lower() for t in tags] if isinstance(tags, list) else False
        )]
        return results
    
    def search_by_media_type(self, media_type: str) -> pd.DataFrame:
        """Filter by media type (image or video)"""
        return self.search_index[self.search_index['media_type'] == media_type]
    
    def combined_search(self, tags: List[str] = None, media_type: Optional[str] = None) -> pd.DataFrame:
        """Search with multiple criteria"""
        results = self.search_index
        
        # Filter by tags
        if tags:
            for tag in tags:
                results = results[results['tags'].apply(
                    lambda t: tag.lower() in [x.lower() for x in t] if isinstance(t, list) else False
                )]
        
        # Filter by media type
        if media_type:
            results = results[results['media_type'] == media_type]
            
        return results

def main():
    parser = argparse.ArgumentParser(description='Search across images and videos')
    parser.add_argument('--tags', nargs='+', help='Tags to search for')
    parser.add_argument('--type', choices=['image', 'video'], help='Media type filter')
    args = parser.parse_args()
    
    search = UnifiedMediaSearch()
    
    if not args.tags and not args.type:
        print("Please provide search criteria (--tags or --type)")
        return
    
    results = search.combined_search(args.tags, args.type)
    
    if len(results) == 0:
        print("No results found.")
    else:
        print(f"Found {len(results)} results:")
        for _, row in results.iterrows():
            print(f"- {row['media_type']}: {row['media_id']} - Tags: {', '.join(row['tags'])}")

if __name__ == "__main__":
    main() 