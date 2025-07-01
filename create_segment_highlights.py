#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create highlight videos from specific timestamp segments.
This script allows you to directly specify time segments to extract from a video
without using AI analysis for segment identification.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import the classes and functions from src modules
from src.models.highlight import HighlightClip
from src.models.font import FontManager
from src.layout.renderer import create_highlight_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

def create_segments_json_template(output_path: str):
    """
    Create a template JSON file for segments.
    
    Args:
        output_path: Path where to save the template file
    """
    template = [
        {
            "start": 10.0,
            "end": 70.0,
            "title": "Introduction and Welcome"
        },
        {
            "start": 120.0,
            "end": 240.0,
            "title": "Main Teaching Point"
        },
        {
            "start": 300.0,
            "end": 420.0,
            "title": "Practical Application"
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created segments template at: {output_path}")
    logger.info("Edit this file with your actual segment timestamps and titles.")

def process_segments(
    input_video: str,
    segments: List[Tuple[float, float, str]],
    args: argparse.Namespace
) -> List[str]:
    """
    Process all segments and create highlight videos.
    
    Args:
        input_video: Path to input video
        segments: List of (start, end, title) tuples
        args: Command line arguments
    
    Returns:
        List of output file paths created
    """
    output_files = []
    
    # Get video info
    try:
        video = VideoFileClip(input_video)
        video_duration = video.duration
        video.close()
    except Exception as e:
        logger.error(f"Failed to load video {input_video}: {e}")
        return []
    
    # Validate segments against video duration
    valid_segments = validate_segments_against_video(segments, video_duration)
    
    if not valid_segments:
        logger.error("No valid segments to process")
        return []
    
    # Process each segment
    input_path = Path(input_video)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for i, (start, end, title) in enumerate(valid_segments, 1):
        logger.info(f"\nProcessing segment {i}/{len(valid_segments)}: {title}")
        
        # Create output filename
        import re
        if args.output_template:
            # Use template with placeholders
            safe_title = re.sub(r'[^\w\-_]', '_', title)
            output_name = args.output_template.format(
                index=i,
                title=safe_title,
                start=int(start),
                end=int(end),
                timestamp=timestamp
            )
        else:
            # Default naming with OS-friendly characters
            safe_title = re.sub(r'[^\w\-_]', '_', title)[:30]  # Replace special chars with underscore
            output_name = f"{input_path.stem}_segment_{i:02d}_{safe_title}_{timestamp}.mp4"
        
        if args.output_dir:
            output_path = Path(args.output_dir) / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path.parent / output_name
        
        output_path = str(output_path)
        
        # Create HighlightClip object
        highlight = HighlightClip(
            start=start,
            end=end,
            score=100.0,  # Max score since user specified
            title=title
        )
        
        try:
            # Create highlight video using existing infrastructure
            create_highlight_video(
                input_path=input_video,
                highlight=highlight,
                segments=None,  # No transcription segments needed
                output_path=output_path,
                main_title=title,
                logo_path=args.logo,
                add_subtitles=False,  # No subtitles without transcription
                vertical_position_ratio=args.vertical_position,
                background_image_path=args.background_image,
                zoom_factor=args.zoom_factor,
                skip_preview=args.auto_confirm,  # Skip preview if auto_confirm is enabled
                skip_logo=args.skip_logo,
                skip_background=args.skip_background,
                skip_service_info=args.skip_service_info
            )
            
            output_files.append(output_path)
            logger.info(f"Created highlight video: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create highlight for segment {i}: {e}")
            continue
    
    return output_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create highlight videos from specific timestamp segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Segment Format Examples:
  Simple:     "10-30,45-75,120-180"
  With titles: "10:30:Introduction;45:75:Main Point;120:180:Conclusion"
  JSON:       '[{"start": 10, "end": 30, "title": "Introduction"}]'
  JSON file:  "segments.json"

Output Template Placeholders:
  {index}     - Segment number (1, 2, 3...)
  {title}     - Segment title (spaces replaced with _)
  {start}     - Start time in seconds
  {end}       - End time in seconds
  {timestamp} - Current timestamp (YYYYMMDD_HHMMSS)
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('segments', help='Segment specifications (see examples above)')
    
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory for highlight videos (default: same as input)')
    parser.add_argument('--output-template', '-t',
                       help='Output filename template (e.g., "highlight_{index}_{title}.mp4")')
    parser.add_argument('--logo', '-l', 
                       default=os.path.join(os.path.dirname(__file__), 'default_assets', 'Waymaker_white_logo_transparent_background.png'),
                       help='Path to logo image file')
    parser.add_argument('--background-image', '-b', 
                       help='Path to banner image for video top')
    parser.add_argument('--skip-logo', action='store_true',
                       help='Skip logo/church name display (default: False)')
    parser.add_argument('--skip-background', action='store_true',
                       help='Skip background/banner image display (default: False)')
    parser.add_argument('--skip-service-info', action='store_true',
                       help='Skip service info display (default: False)')
    parser.add_argument('--zoom-factor', '-z', type=float, default=2.0,
                       help='Zoom factor for the video (default: 2.0)')
    parser.add_argument('--vertical-position', '-v', type=float, default=0.67,
                       help='Vertical position ratio (default: 0.67)')
    parser.add_argument('--create-template', action='store_true',
                       help='Create a JSON template file for segments')
    parser.add_argument('--template-output', default='segments_template.json',
                       help='Output path for template file (default: segments_template.json)')
    parser.add_argument('--auto-confirm', '-y', action='store_true',
                       help='Skip preview confirmations and process automatically')
    parser.add_argument('--manual-title', nargs='+',
                       help='Manual titles for segments (one title per segment, in order)')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        create_segments_json_template(args.template_output)
        return
    
    # Validate input video
    if not os.path.isfile(args.input_video):
        logger.error(f"Input video file not found: {args.input_video}")
        sys.exit(1)
    
    # Check logo file
    if args.logo and not os.path.exists(args.logo):
        logger.warning(f"Logo file not found: {args.logo}")
        args.logo = None
    
    # Parse segments
    logger.info(f"Parsing segments: {args.segments}")
    segments = parse_segments(args.segments)
    
    if not segments:
        logger.error("No valid segments found")
        sys.exit(1)
    
    # Apply manual titles if provided
    if args.manual_title:
        if len(args.manual_title) != len(segments):
            logger.error(f"Number of manual titles ({len(args.manual_title)}) must match number of segments ({len(segments)})")
            sys.exit(1)
        
        # Override titles with manual titles
        segments = [(start, end, manual_title) for (start, end, _), manual_title in zip(segments, args.manual_title)]
        logger.info("Applied manual titles to segments:")
        for i, (start, end, title) in enumerate(segments, 1):
            duration = end - start
            logger.info(f"  {i}. {start:.1f}s - {end:.1f}s ({duration:.1f}s): {title}")
    
    # Process segments
    logger.info(f"Processing {len(segments)} segments from video: {args.input_video}")
    output_files = process_segments(args.input_video, segments, args)
    
    # Summary
    if output_files:
        logger.info(f"\nSuccessfully created {len(output_files)} highlight videos:")
        for output_file in output_files:
            logger.info(f"  - {output_file}")
    else:
        logger.error("No highlight videos were created")
        sys.exit(1)

if __name__ == "__main__":
    main()