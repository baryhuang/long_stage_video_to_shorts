#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple segment stitching script that extracts and concatenates video segments
without any processing (no zoom, no face tracking, no layout changes).
Just cuts the segments and stitches them together.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_segments(segments_input: str) -> List[Tuple[float, float, str]]:
    """
    Parse segment input string into list of (start, end, title) tuples.
    
    Supports multiple formats:
    1. "start1-end1,start2-end2" (e.g., "10-30,45-75")
    2. "start1:end1:title1;start2:end2:title2" (e.g., "10:30:Introduction;45:75:Main Point")
    3. JSON format: [{"start": 10, "end": 30, "title": "Introduction"}]
    4. File path to JSON file containing segments
    
    Args:
        segments_input: String containing segment information
        
    Returns:
        List of (start_time, end_time, title) tuples
    """
    segments = []
    
    # Check if it's a file path
    if os.path.isfile(segments_input):
        logger.info(f"Loading segments from file: {segments_input}")
        try:
            with open(segments_input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        start = float(item.get('start', 0))
                        end = float(item.get('end', start + 60))
                        title = item.get('title', f'Segment {i+1}')
                        segments.append((start, end, title))
                    else:
                        logger.warning(f"Invalid segment format in file: {item}")
            else:
                logger.error("JSON file should contain a list of segment objects")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load segments from file: {e}")
            return []
    
    # Try to parse as JSON string
    elif segments_input.strip().startswith('['):
        try:
            data = json.loads(segments_input)
            for i, item in enumerate(data):
                start = float(item.get('start', 0))
                end = float(item.get('end', start + 60))
                title = item.get('title', f'Segment {i+1}')
                segments.append((start, end, title))
        except Exception as e:
            logger.error(f"Failed to parse JSON segments: {e}")
            return []
    
    # Parse colon-semicolon format: "start:end:title;start:end:title"
    elif ':' in segments_input and ';' in segments_input:
        segment_parts = segments_input.split(';')
        for i, part in enumerate(segment_parts):
            parts = part.strip().split(':')
            if len(parts) >= 2:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    title = parts[2] if len(parts) > 2 else f'Segment {i+1}'
                    segments.append((start, end, title))
                except ValueError as e:
                    logger.warning(f"Invalid segment format: {part} - {e}")
    
    # Parse simple comma format: "start1-end1,start2-end2"
    elif ',' in segments_input or '-' in segments_input:
        if ',' in segments_input:
            segment_parts = segments_input.split(',')
        else:
            segment_parts = [segments_input]
            
        for i, part in enumerate(segment_parts):
            if '-' in part:
                try:
                    start_str, end_str = part.strip().split('-', 1)
                    start = float(start_str)
                    end = float(end_str)
                    title = f'Segment {i+1}'
                    segments.append((start, end, title))
                except ValueError as e:
                    logger.warning(f"Invalid segment format: {part} - {e}")
    
    else:
        logger.error(f"Unable to parse segments format: {segments_input}")
        logger.info("Supported formats:")
        logger.info("  1. Simple: '10-30,45-75'")
        logger.info("  2. With titles: '10:30:Introduction;45:75:Main Point'")
        logger.info("  3. JSON: '[{\"start\": 10, \"end\": 30, \"title\": \"Introduction\"}]'")
        logger.info("  4. JSON file path: 'segments.json'")
        return []
    
    # Validate segments
    valid_segments = []
    for start, end, title in segments:
        if start >= end:
            logger.warning(f"Invalid segment: start ({start}) >= end ({end}), skipping")
            continue
        if start < 0:
            logger.warning(f"Invalid segment: start time ({start}) < 0, adjusting to 0")
            start = 0
        valid_segments.append((start, end, title))
    
    logger.info(f"Parsed {len(valid_segments)} valid segments:")
    for i, (start, end, title) in enumerate(valid_segments, 1):
        duration = end - start
        logger.info(f"  {i}. {start:.1f}s - {end:.1f}s ({duration:.1f}s): {title}")
    
    return valid_segments

def validate_segments_against_video(segments: List[Tuple[float, float, str]], 
                                  video_duration: float) -> List[Tuple[float, float, str]]:
    """
    Validate segments against video duration and adjust if necessary.
    
    Args:
        segments: List of (start, end, title) tuples
        video_duration: Duration of the video in seconds
    
    Returns:
        List of valid segments, adjusted if necessary
    """
    valid_segments = []
    
    for start, end, title in segments:
        # Adjust end time if it exceeds video duration
        if end > video_duration:
            logger.warning(f"Segment '{title}' end time ({end:.1f}s) exceeds video duration ({video_duration:.1f}s)")
            end = video_duration
            logger.info(f"Adjusted end time to {end:.1f}s")
        
        # Skip segments that start after video ends
        if start >= video_duration:
            logger.warning(f"Segment '{title}' starts after video ends, skipping")
            continue
        
        # Ensure minimum duration
        if end - start < 1.0:
            logger.warning(f"Segment '{title}' duration ({end-start:.1f}s) too short, skipping")
            continue
            
        valid_segments.append((start, end, title))
    
    return valid_segments

def stitch_segments(
    input_video: str,
    segments: List[Tuple[float, float, str]],
    output_path: str
) -> bool:
    """
    Extract segments from video and stitch them together.
    
    Args:
        input_video: Path to input video
        segments: List of (start, end, title) tuples
        output_path: Path to output video
    
    Returns:
        bool: True if successful
    """
    try:
        # Get video info
        video = VideoFileClip(input_video)
        video_duration = video.duration
        fps = video.fps
        video.close()
        
        logger.info(f"Input video: {input_video}")
        logger.info(f"Duration: {video_duration:.2f}s, FPS: {fps}")
        
        # Validate segments against video duration
        valid_segments = validate_segments_against_video(segments, video_duration)
        
        if not valid_segments:
            logger.error("No valid segments to process")
            return False
        
        # Extract each segment
        logger.info(f"Extracting {len(valid_segments)} segments...")
        clips = []
        
        # Load video once
        video = VideoFileClip(input_video)
        
        for i, (start, end, title) in enumerate(valid_segments, 1):
            logger.info(f"Extracting segment {i}/{len(valid_segments)}: {title} ({start:.1f}s - {end:.1f}s)")
            
            # Extract segment
            segment_clip = video.subclip(start, end)
            clips.append(segment_clip)
        
        # Concatenate all segments
        logger.info("Stitching segments together...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Calculate total duration
        total_duration = sum(end - start for start, end, _ in valid_segments)
        logger.info(f"Total stitched duration: {total_duration:.2f}s")
        
        # Write output
        logger.info(f"Writing output to: {output_path}")
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )
        
        # Clean up
        final_clip.close()
        for clip in clips:
            clip.close()
        video.close()
        
        logger.info("Stitching completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to stitch segments: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Stitch video segments together without any processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Segment Format Examples:
  Simple:     "10-30,45-75,120-180"
  With titles: "10:30:Introduction;45:75:Main Point;120:180:Conclusion"
  JSON:       '[{"start": 10, "end": 30, "title": "Introduction"}]'
  JSON file:  "segments.json"
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('segments', help='Segment specifications (see examples above)')
    parser.add_argument('--output', '-o', help='Output video path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.isfile(args.input_video):
        logger.error(f"Input video file not found: {args.input_video}")
        sys.exit(1)
    
    # Parse segments
    logger.info(f"Parsing segments: {args.segments}")
    segments = parse_segments(args.segments)
    
    if not segments:
        logger.error("No valid segments found")
        sys.exit(1)
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input_video)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = str(input_path.with_name(f"{input_path.stem}_stitched_{timestamp}.mp4"))
        logger.info(f"Output path: {args.output}")
    
    # Stitch segments
    if stitch_segments(args.input_video, segments, args.output):
        logger.info(f"Successfully created stitched video: {args.output}")
    else:
        logger.error("Failed to create stitched video")
        sys.exit(1)

if __name__ == "__main__":
    main()