#!/usr/bin/env python3
# export_reports.py - Generate reports of identified individuals

import os
import json
import argparse
import csv
from pathlib import Path
import pandas as pd
from datetime import datetime

class MediaExporter:
    """Export media identification data in various formats"""
    
    def __init__(self, output_dir="exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data sources
        self.video_tags_path = "video_tags.json"
        self.unique_video_tags_path = "unique_tags_by_video.json"
        self.face_metadata_path = "detected_faces/face_metadata.json"
        
        # Load data
        self.video_tags = self._load_json(self.video_tags_path, {})
        self.unique_video_tags = self._load_json(self.unique_video_tags_path, {})
        self.face_metadata = self._load_json(self.face_metadata_path, {})
        
        # Timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _load_json(self, path, default=None):
        """Load JSON file with error handling"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return default
    
    def export_video_tags_csv(self):
        """Export video tags to CSV format"""
        if not self.video_tags:
            print("No video tags data available")
            return None
            
        # Prepare data for CSV
        rows = []
        
        for video_id, frames in self.video_tags.items():
            for frame_id, frame_data in frames.items():
                for tag in frame_data.get('tags', []):
                    rows.append({
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'tag': tag,
                        'confidence': frame_data.get('confidence', {}).get(tag, 0),
                    })
        
        if not rows:
            print("No tag data to export")
            return None
            
        # Convert to DataFrame and save
        df = pd.DataFrame(rows)
        output_path = self.output_dir / f"video_tags_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Exported video tags to {output_path}")
        return output_path
    
    def export_unique_tags_csv(self):
        """Export unique tags per video to CSV format"""
        if not self.unique_video_tags:
            print("No unique video tags data available")
            return None
            
        # Prepare data for CSV
        rows = []
        
        for video_id, tags in self.unique_video_tags.items():
            for tag in tags:
                rows.append({
                    'video_id': video_id,
                    'tag': tag
                })
        
        if not rows:
            print("No unique tag data to export")
            return None
            
        # Convert to DataFrame and save
        df = pd.DataFrame(rows)
        output_path = self.output_dir / f"unique_video_tags_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Exported unique video tags to {output_path}")
        return output_path
    
    def export_face_metadata_csv(self):
        """Export face metadata to CSV format"""
        if not self.face_metadata:
            print("No face metadata available")
            return None
            
        # Prepare data for CSV
        rows = []
        
        for face_id, metadata in self.face_metadata.items():
            # Extract key fields, handling missing values
            row = {
                'face_id': face_id,
                'tag': metadata.get('tag', 'unknown'),
                'source_image': metadata.get('source_image', ''),
                'confidence': metadata.get('confidence', 0),
                'x': metadata.get('bbox', [0, 0, 0, 0])[0] if 'bbox' in metadata else 0,
                'y': metadata.get('bbox', [0, 0, 0, 0])[1] if 'bbox' in metadata else 0,
                'width': metadata.get('bbox', [0, 0, 0, 0])[2] if 'bbox' in metadata else 0,
                'height': metadata.get('bbox', [0, 0, 0, 0])[3] if 'bbox' in metadata else 0,
                'blur_score': metadata.get('blur_score', 0),
                'cluster_id': metadata.get('cluster_id', -1)
            }
            rows.append(row)
        
        if not rows:
            print("No face metadata to export")
            return None
            
        # Convert to DataFrame and save
        df = pd.DataFrame(rows)
        output_path = self.output_dir / f"face_metadata_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Exported face metadata to {output_path}")
        return output_path
    
    def export_tag_summary(self):
        """Export a summary of all tags and their occurrences"""
        # Collect all tags from videos
        video_tag_counts = {}
        for video_id, tags in self.unique_video_tags.items():
            for tag in tags:
                if tag not in video_tag_counts:
                    video_tag_counts[tag] = 0
                video_tag_counts[tag] += 1
        
        # Collect all tags from faces
        face_tag_counts = {}
        for face_id, metadata in self.face_metadata.items():
            if 'tag' in metadata and metadata['tag']:
                tag = metadata['tag']
                if tag not in face_tag_counts:
                    face_tag_counts[tag] = 0
                face_tag_counts[tag] += 1
        
        # Combine all unique tags
        all_tags = set(list(video_tag_counts.keys()) + list(face_tag_counts.keys()))
        
        # Create summary rows
        rows = []
        for tag in sorted(all_tags):
            rows.append({
                'tag': tag,
                'video_occurrences': video_tag_counts.get(tag, 0),
                'face_occurrences': face_tag_counts.get(tag, 0),
                'total_occurrences': video_tag_counts.get(tag, 0) + face_tag_counts.get(tag, 0)
            })
        
        # Sort by total occurrences
        rows.sort(key=lambda x: x['total_occurrences'], reverse=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(rows)
        output_path = self.output_dir / f"tag_summary_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Exported tag summary to {output_path}")
        return output_path
    
    def export_all(self):
        """Export all available data formats"""
        self.export_video_tags_csv()
        self.export_unique_tags_csv()
        self.export_face_metadata_csv()
        self.export_tag_summary()
        
        # Create a combined Excel file with multiple sheets
        excel_path = self.output_dir / f"complete_export_{self.timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path) as writer:
            # Video tags sheet
            if self.video_tags:
                rows = []
                for video_id, frames in self.video_tags.items():
                    for frame_id, frame_data in frames.items():
                        for tag in frame_data.get('tags', []):
                            rows.append({
                                'video_id': video_id,
                                'frame_id': frame_id,
                                'tag': tag,
                                'confidence': frame_data.get('confidence', {}).get(tag, 0),
                            })
                if rows:
                    pd.DataFrame(rows).to_excel(writer, sheet_name='Video Tags', index=False)
            
            # Unique video tags sheet
            if self.unique_video_tags:
                rows = []
                for video_id, tags in self.unique_video_tags.items():
                    for tag in tags:
                        rows.append({
                            'video_id': video_id,
                            'tag': tag
                        })
                if rows:
                    pd.DataFrame(rows).to_excel(writer, sheet_name='Unique Video Tags', index=False)
            
            # Face metadata sheet
            if self.face_metadata:
                rows = []
                for face_id, metadata in self.face_metadata.items():
                    row = {
                        'face_id': face_id,
                        'tag': metadata.get('tag', 'unknown'),
                        'source_image': metadata.get('source_image', ''),
                        'confidence': metadata.get('confidence', 0),
                        'blur_score': metadata.get('blur_score', 0),
                        'cluster_id': metadata.get('cluster_id', -1)
                    }
                    rows.append(row)
                if rows:
                    pd.DataFrame(rows).to_excel(writer, sheet_name='Face Metadata', index=False)
            
            # Tag summary sheet
            video_tag_counts = {}
            for video_id, tags in self.unique_video_tags.items():
                for tag in tags:
                    if tag not in video_tag_counts:
                        video_tag_counts[tag] = 0
                    video_tag_counts[tag] += 1
            
            face_tag_counts = {}
            for face_id, metadata in self.face_metadata.items():
                if 'tag' in metadata and metadata['tag']:
                    tag = metadata['tag']
                    if tag not in face_tag_counts:
                        face_tag_counts[tag] = 0
                    face_tag_counts[tag] += 1
            
            all_tags = set(list(video_tag_counts.keys()) + list(face_tag_counts.keys()))
            rows = []
            for tag in sorted(all_tags):
                rows.append({
                    'tag': tag,
                    'video_occurrences': video_tag_counts.get(tag, 0),
                    'face_occurrences': face_tag_counts.get(tag, 0),
                    'total_occurrences': video_tag_counts.get(tag, 0) + face_tag_counts.get(tag, 0)
                })
            
            rows.sort(key=lambda x: x['total_occurrences'], reverse=True)
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='Tag Summary', index=False)
        
        print(f"Exported combined Excel file to {excel_path}")
        return excel_path

def main():
    parser = argparse.ArgumentParser(description='Export media identification data')
    parser.add_argument('--output', default='exports', help='Output directory for exports')
    parser.add_argument('--format', choices=['csv', 'excel', 'all'], default='all', 
                      help='Export format (csv, excel, or all)')
    args = parser.parse_args()
    
    exporter = MediaExporter(output_dir=args.output)
    
    if args.format == 'csv' or args.format == 'all':
        exporter.export_video_tags_csv()
        exporter.export_unique_tags_csv()
        exporter.export_face_metadata_csv()
        exporter.export_tag_summary()
    
    if args.format == 'excel' or args.format == 'all':
        exporter.export_all()

if __name__ == "__main__":
    main() 