#!/usr/bin/env python3
# tag_statistics_dashboard.py - Visualize tag statistics across media collection

import os
import json
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

class TagStatisticsDashboard:
    """Generate statistics and visualizations for face and video tags"""
    
    def __init__(self, output_dir="statistics"):
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
    
    def analyze_video_tags(self):
        """Analyze video tag distribution"""
        # Count tag occurrences across all videos
        tag_counter = Counter()
        
        for video_id, tags in self.unique_video_tags.items():
            for tag in tags:
                tag_counter[tag] += 1
        
        # Get top tags
        top_tags = tag_counter.most_common(20)
        
        # Create dataframe for analysis
        df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
        
        return df
    
    def analyze_face_tags(self):
        """Analyze face tag distribution"""
        # Count tag occurrences across all faces
        tag_counter = Counter()
        
        for face_id, metadata in self.face_metadata.items():
            if 'tag' in metadata and metadata['tag']:
                tag = metadata['tag']
                tag_counter[tag] += 1
        
        # Get top tags
        top_tags = tag_counter.most_common(20)
        
        # Create dataframe for analysis
        df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
        
        return df
    
    def analyze_tag_by_video(self):
        """Analyze tag distribution by video"""
        # Create a dict of video -> tag counts
        video_tag_counts = defaultdict(Counter)
        
        for video_id, tags in self.unique_video_tags.items():
            for tag in tags:
                video_tag_counts[video_id][tag] += 1
        
        # Convert to dataframe for easier analysis
        records = []
        for video_id, tag_counter in video_tag_counts.items():
            for tag, count in tag_counter.items():
                records.append({
                    'Video': video_id,
                    'Tag': tag,
                    'Count': count
                })
        
        df = pd.DataFrame(records)
        return df
    
    def plot_top_tags(self, save=True):
        """Plot top tags across all media"""
        # Get tag statistics
        video_tags_df = self.analyze_video_tags()
        face_tags_df = self.analyze_face_tags()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot video tags
        if not video_tags_df.empty and len(video_tags_df) > 0:
            video_tags_df.sort_values('Count', ascending=True).tail(10).plot(
                kind='barh', x='Tag', y='Count', ax=ax1, color='skyblue'
            )
            ax1.set_title('Top 10 Video Tags')
            ax1.set_xlabel('Count')
            ax1.set_ylabel('Tag')
        else:
            ax1.text(0.5, 0.5, 'No video tag data available', 
                     horizontalalignment='center', verticalalignment='center')
        
        # Plot face tags
        if not face_tags_df.empty and len(face_tags_df) > 0:
            face_tags_df.sort_values('Count', ascending=True).tail(10).plot(
                kind='barh', x='Tag', y='Count', ax=ax2, color='lightgreen'
            )
            ax2.set_title('Top 10 Face Tags')
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Tag')
        else:
            ax2.text(0.5, 0.5, 'No face tag data available', 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'top_tags.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir / 'top_tags.png'}")
        
        return fig
    
    def plot_tag_distribution_by_video(self, top_n=5, save=True):
        """Plot tag distribution across videos"""
        # Get tag by video data
        df = self.analyze_tag_by_video()
        
        if df.empty:
            print("No data available for tag distribution by video")
            return None
        
        # Get top N tags overall
        top_tags = df.groupby('Tag')['Count'].sum().nlargest(top_n).index.tolist()
        
        # Filter dataframe to only include top tags
        df_top = df[df['Tag'].isin(top_tags)]
        
        # Pivot the data for plotting
        pivot_df = df_top.pivot_table(
            index='Video', 
            columns='Tag', 
            values='Count',
            fill_value=0
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Distribution of Top {top_n} Tags Across Videos')
        ax.set_xlabel('Video')
        ax.set_ylabel('Count')
        ax.legend(title='Tag')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'tag_distribution_by_video.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir / 'tag_distribution_by_video.png'}")
        
        return fig
    
    def generate_statistics_report(self):
        """Generate comprehensive statistics report as JSON"""
        # Count total videos, images, faces
        total_videos = len(self.unique_video_tags)
        
        # Count unique images from face metadata
        unique_images = set()
        total_faces = 0
        tagged_faces = 0
        
        for face_id, metadata in self.face_metadata.items():
            total_faces += 1
            if 'source_image' in metadata:
                unique_images.add(metadata['source_image'])
            if 'tag' in metadata and metadata['tag'] and metadata['tag'].lower() != 'unknown':
                tagged_faces += 1
        
        # Count unique tags
        video_unique_tags = set()
        for tags in self.unique_video_tags.values():
            video_unique_tags.update(tags)
            
        face_unique_tags = set()
        for metadata in self.face_metadata.values():
            if 'tag' in metadata and metadata['tag']:
                face_unique_tags.add(metadata['tag'])
        
        # Generate report
        report = {
            "total_videos": total_videos,
            "total_unique_images": len(unique_images),
            "total_faces_detected": total_faces,
            "total_faces_tagged": tagged_faces,
            "tagging_completion_rate": round(tagged_faces / total_faces * 100 if total_faces > 0 else 0, 2),
            "unique_video_tags": len(video_unique_tags),
            "unique_face_tags": len(face_unique_tags),
            "video_tags_list": sorted(list(video_unique_tags)),
            "face_tags_list": sorted(list(face_unique_tags))
        }
        
        # Save report
        with open(self.output_dir / 'statistics_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Saved statistics report to {self.output_dir / 'statistics_report.json'}")
        
        return report
    
    def generate_all(self):
        """Generate all statistics and plots"""
        self.generate_statistics_report()
        self.plot_top_tags()
        self.plot_tag_distribution_by_video()
        
        print("All statistics and visualizations generated successfully")

def main():
    parser = argparse.ArgumentParser(description='Generate tag statistics dashboard')
    parser.add_argument('--output', default='statistics', help='Output directory for statistics and plots')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving them')
    args = parser.parse_args()
    
    dashboard = TagStatisticsDashboard(output_dir=args.output)
    
    if args.show:
        dashboard.plot_top_tags(save=False)
        dashboard.plot_tag_distribution_by_video(save=False)
        plt.show()
    else:
        dashboard.generate_all()

if __name__ == "__main__":
    main() 