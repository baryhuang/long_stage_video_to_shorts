#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Highlight video generator that extracts engaging segments from longer videos.
The output is in 9:16 portrait format with traditional Chinese titles and subtitles.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from dotenv import load_dotenv
import assemblyai as aai
from anthropic import Anthropic
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip, ColorClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from tqdm import tqdm
import zhconv  # For converting between Chinese variants
import requests
import openai
from PIL import Image, ImageDraw, ImageFont
import jieba

# Import the classes and functions from src modules
from src.models.transcription import TranscriptionSegment
from src.models.highlight import HighlightClip
from src.models.font import FontManager
from src.transcription.transcriber import transcribe_video, create_text_transcript
from src.video.processor import track_and_zoom_video
from src.highlights.analyzer import identify_highlights
from src.highlights.titles import generate_titles
from src.video.preview import create_zoom_preview
from src.layout.preview import create_layout_preview
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

# Get API keys from environment
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASSEMBLY_API_KEY:
    logger.error("AssemblyAI API key not found. Please set the ASSEMBLY_API_KEY environment variable.")
    sys.exit(1)

if not ANTHROPIC_API_KEY:
    logger.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    sys.exit(1)

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Initialize API clients
aai.settings.api_key = ASSEMBLY_API_KEY
# anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.Client(api_key=OPENAI_API_KEY)

def save_state(video_path: str, state_name: str, data: Any):
    """Save state data to a JSON file"""
    video_path_obj = Path(video_path)
    state_file = video_path_obj.with_name(f"{video_path_obj.stem}_{state_name}.state.json")
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {state_name} state to {state_file}")

def load_state(video_path: str, state_name: str) -> Optional[Any]:
    """Load state data from a JSON file if it exists"""
    video_path_obj = Path(video_path)
    state_file = video_path_obj.with_name(f"{video_path_obj.stem}_{state_name}.state.json")
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {state_name} state from {state_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load {state_name} state: {e}")
    return None

def process_video_with_zoom_and_tracking(clip: VideoFileClip, zoom_factor: float = 2.0, vertical_position_ratio: float = 0.5) -> VideoFileClip:
    """Use track_and_zoom_video from the video processor module.
    This function is kept for backwards compatibility."""
    return track_and_zoom_video(clip, 
                               min_zoom=zoom_factor, 
                               max_zoom=zoom_factor,
                               smoothing_window=15,
                               vertical_position_ratio=vertical_position_ratio)

def create_full_video_segment(video_path: str) -> Dict[str, Any]:
    """
    Create a segment that covers the entire video.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary containing segment information for the full video
    """
    # Get video duration
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()
    
    # Create a segment covering the entire video
    return {
        "segments": [
            {
                "start_time": 0.0,
                "end_time": duration,
                "content": "完整影片",  # "Full video" in Traditional Chinese
                "reason": "使用完整影片進行處理"  # "Processing the entire video" in Traditional Chinese
            }
        ],
        "theme": "完整影片處理"  # "Full video processing" in Traditional Chinese
    }

def find_supported_files(folder_path: str) -> List[str]:
    """
    Recursively find all supported audio and video files in a folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of paths to supported files
    """
    supported_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm',  # Video formats
        '.mp3', '.wav', '.m4a', '.aac', '.ogg'    # Audio formats
    }
    
    found_files = []
    
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
        
        # Sort files for consistent processing order
        found_files.sort()
        
        if not found_files:
            logger.warning(f"No supported files found in {folder_path}")
        else:
            logger.info(f"Found {len(found_files)} supported files in {folder_path}")
            for file in found_files:
                logger.info(f"- {file}")
    
    except Exception as e:
        logger.error(f"Error searching folder {folder_path}: {e}")
        return []
    return found_files

def create_highlight_video(
    input_path: str, 
    highlight: HighlightClip, 
    segments: Optional[List[TranscriptionSegment]],
    output_path: str,
    main_title: str,
    logo_path: Optional[str] = None,
    add_subtitles: bool = False,
    vertical_position_ratio: float = 0.67,
    background_image_path: Optional[str] = None,
    zoom_factor: float = 2.0
):
    """
    Create highlight video in 9:16 format with titles and subtitles, styled like a church service video
    First creates a preview for confirmation, then proceeds with video creation if approved.
    Uses the segment title as the main title text.
    The main video section will be in 1:1 ratio.
    
    Args:
        input_path: Path to input video
        highlight: HighlightClip object with timing information
        segments: List of transcription segments (optional, None for full video mode)
        output_path: Path to output video
        main_title: Ignored as we'll use highlight.title instead
        logo_path: Path to logo image file (optional)
        add_subtitles: Whether to add subtitles to the video (default: False)
        vertical_position_ratio: Ratio for vertical positioning (default: 0.67 for 2/3 from top)
        background_image_path: Path to background image file (optional)
        zoom_factor: Factor to zoom the video (default: 2.0 for 200% zoom)
    """
    # Create zoom preview first
    zoom_preview_path = str(Path(output_path).with_suffix('.zoom_preview.jpg'))
    preview_timestamp = highlight.start + (highlight.end - highlight.start) / 2  # Use middle frame for preview
    
    # Show zoom preview and get confirmation
    if not create_zoom_preview(
        input_path=input_path,
        timestamp=preview_timestamp,
        zoom_factor=zoom_factor,  # Use parameter instead of fixed value
        vertical_position_ratio=vertical_position_ratio,
        output_preview_path=zoom_preview_path
    ):
        logger.info("Video creation cancelled by user after zoom preview")
        return
    
    # Create layout preview
    layout_preview_path = str(Path(output_path).with_suffix('.layout_preview.jpg'))
    if not create_layout_preview(
        input_path=input_path,
        timestamp=preview_timestamp,
        main_title=highlight.title,
        output_preview_path=layout_preview_path,
        logo_path=logo_path,
        segments=segments,  # This can be None now
        highlight=highlight,
        zoom_factor=zoom_factor,
        vertical_position_ratio=vertical_position_ratio
    ):
        logger.info("Video creation cancelled by user after layout preview")
        return

    logger.info(f"Creating highlight video from {highlight.start:.2f}s to {highlight.end:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    
    # Load video clip and apply zoom and tracking
    video = VideoFileClip(input_path).subclip(highlight.start, highlight.end)
    logger.info(f"Original video dimensions: {video.size[0]}x{video.size[1]} (width x height)")
    
    # Apply zoom and tracking with vertical position ratio
    logger.info(f"Applying {zoom_factor*100}% zoom and horizontal person tracking with vertical position ratio {vertical_position_ratio}...")
    processed_clip = track_and_zoom_video(
        video, 
        min_zoom=zoom_factor, 
        max_zoom=zoom_factor,
        smoothing_window=15,
        vertical_position_ratio=vertical_position_ratio
    )
    
    # Calculate dimensions for 9:16 portrait format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Calculate dimensions for video section (50% of height)
    video_section_width = output_width
    video_section_height = int(output_height * 0.5)  # Exactly 50% of the total height
    logger.info(f"Video section dimensions: {video_section_width}x{video_section_height} (width x height)")
    
    # Position the video section in the center vertically
    video_section_y = (output_height - video_section_height) // 2
    
    # Text position constants
    title_margin = 25
    church_name_height = 100
    main_title_height = 100
    service_info_height = 50
    
    # Calculate text positions above the video section
    top_text_height = church_name_height + main_title_height + service_info_height + title_margin*2
    
    # Position church name at the top with margin
    church_name_y = title_margin
    
    # If that would overlap with video, position church name so all text fits above video
    if church_name_y + top_text_height > video_section_y:
        # Position church name so that all text elements end right before video section
        church_name_y = max(title_margin, video_section_y - top_text_height + title_margin)
    
    main_title_y = church_name_y + church_name_height + main_title_height//2
    # Add more space between main title and service info (increase by 30px)
    service_info_y = main_title_y + main_title_height//2 + service_info_height//2 + 30
    
    # Resize video while maintaining aspect ratio to fit the desired size
    orig_height, orig_width = processed_clip.size[1], processed_clip.size[0]
    
    # Calculate the scaled size based on 50% of the output height
    scale_factor = video_section_height / orig_height
    scaled_width = int(orig_width * scale_factor)
    scaled_height = video_section_height  # Always 50% of the total height
    
    # Resize the video
    processed_clip = processed_clip.resize(width=scaled_width, height=scaled_height)
    logger.info(f"Resized video dimensions: {scaled_width}x{scaled_height} (width x height)")
    
    # Calculate position to center the video horizontally
    x_offset = (video_section_width - scaled_width) // 2
    y_offset = 0  # No vertical offset needed as we're using exactly 50% height
    
    # Create black background for the 9:16 format
    black_bg = ColorClip(size=(output_width, output_height), color=(0, 0, 0))
    black_bg = black_bg.set_duration(video.duration)
    
    # Position the video in the center vertically
    video_y = video_section_y
    
    # If video is wider than output width, we'll need to crop it
    if scaled_width > output_width:
        # Create a crop effect to show only the center portion of the wider video
        x_crop_offset = (scaled_width - output_width) // 2
        processed_clip = processed_clip.crop(x1=x_crop_offset, y1=0, x2=x_crop_offset+output_width, y2=scaled_height)
        video_x = 0
        logger.info(f"Video wider than screen, cropping {x_crop_offset}px from each side")
    else:
        # Center the video if it's narrower than the output width
        video_x = x_offset if x_offset > 0 else 0
    
    video_pos = (video_x if video_x != 0 else 'center', video_y)
    logger.info(f"Video position: x={video_x if video_x != 0 else 'center'}, y={video_y} (distance from top)")
    
    # Create a clip for the video section border to make it clearly visible
    border_color = (50, 50, 50)  # Dark gray border
    video_border = ColorClip(size=(video_section_width + 4, video_section_height + 4), color=border_color)
    video_border = video_border.set_duration(video.duration)
    video_border_pos = ('center', video_section_y - 2)  # Position the border with 2px on each side
    
    # Initialize clips list with background, border, and video
    clips_to_combine = [
        black_bg, 
        video_border.set_position(video_border_pos),
        processed_clip.set_position(video_pos)
    ]
    
    # Position banner image and text elements relative to the video section
    banner_height = 0  # Default to 0 if no banner image
    
    # Add banner image at the top if provided
    banner_clip = None
    if background_image_path and os.path.exists(background_image_path):
        try:
            # Load the banner image
            logger.info(f"Loading banner image from {background_image_path}")
            banner_img = ImageClip(background_image_path)
            
            # Calculate banner size (full width, proportional height)
            banner_aspect = banner_img.size[0] / banner_img.size[1]
            banner_width = output_width
            banner_height = int(banner_width / banner_aspect)
            
            # Limit banner height to a reasonable size (max 15% of screen height)
            max_banner_height = int(output_height * 0.15)
            if banner_height > max_banner_height:
                banner_height = max_banner_height
                banner_width = int(banner_height * banner_aspect)
                # Recenter if width is reduced
                banner_x = (output_width - banner_width) // 2
            else:
                banner_x = 0
            
            # Resize and create banner clip
            banner_clip = banner_img.resize(newsize=(banner_width, banner_height))
            banner_clip = banner_clip.set_position((banner_x, 0))  # Position at top
            banner_clip = banner_clip.set_duration(video.duration)
            
            logger.info(f"Added banner image with dimensions: {banner_width}x{banner_height}")
        except Exception as e:
            logger.warning(f"Failed to load banner image: {e}")
            banner_height = 0  # Reset if failed
    
    # Add banner to clips if loaded successfully
    if banner_clip:
        clips_to_combine.append(banner_clip)
    
    # Add logo or church name with increased size
    if logo_path and os.path.exists(logo_path):
        try:
            # Create logo clip
            logo_clip = ImageClip(logo_path)
            
            # Calculate logo size with increased height
            logo_height = church_name_height
            logo_width = int(logo_clip.size[0] * (logo_height / logo_clip.size[1]))
            
            # Allow logo to be wider, up to 90% of output width
            max_logo_width = int(output_width * 0.9)
            if logo_width > max_logo_width:
                logo_width = max_logo_width
                logo_height = int(logo_clip.size[1] * (logo_width / logo_clip.size[0]))
            
            # Resize and position logo
            logo_clip = logo_clip.resize(newsize=(logo_width, logo_height))
            logo_x = (output_width - logo_width) // 2
            logo_clip = logo_clip.set_position((logo_x, church_name_y))
            logo_clip = logo_clip.set_duration(video.duration)
            
            clips_to_combine.append(logo_clip)
            logger.info(f"Added logo clip: {logo_width}x{logo_height} at position ({logo_x}, {church_name_y})")
        except Exception as e:
            logger.warning(f"Failed to add logo: {e}")
            # Fallback to text
            church_name_clip = TextClip(
                "開路者教會 WAYMAKER CHURCH",
                fontsize=50,
                color='white',
                font=font_path,
                size=(output_width - 2*title_margin, church_name_height)
            ).set_position(('center', church_name_y))
            church_name_clip = church_name_clip.set_duration(video.duration)
            clips_to_combine.append(church_name_clip)
            logger.info("Added church name text clip as fallback")
    else:
        # Use text for church name
        church_name_clip = TextClip(
            "開路者教會 WAYMAKER CHURCH",
            fontsize=50,
            color='white',
            font=font_path,
            size=(output_width - 2*title_margin, church_name_height)
        ).set_position(('center', church_name_y))
        church_name_clip = church_name_clip.set_duration(video.duration)
        clips_to_combine.append(church_name_clip)
        logger.info("Added church name text clip")
    
    # Add main title with multi-line support
    title_text = highlight.title
    title_fontsize = 80
    
    # Check if title needs to be split into multiple lines
    title_width_estimate = len(title_text) * title_fontsize * 0.6  # Rough estimate
    available_width = output_width - 2 * title_margin
    
    if title_width_estimate > available_width:
        # Try to find a natural break point
        break_points = []
        for i, char in enumerate(title_text):
            if char in '。，！？,.!? ':
                break_points.append(i)
        
        if break_points and len(title_text) > 10:
            # Find the best break point near the middle
            middle_idx = len(title_text) // 2
            best_break = min(break_points, key=lambda x: abs(x - middle_idx))
            
            # Split the title into two lines
            line1 = title_text[:best_break+1]
            line2 = title_text[best_break+1:]
            
            # Create two separate text clips
            line1_clip = TextClip(
                line1,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height//2)
            ).set_position(('center', main_title_y - title_fontsize//2 - 5))
            
            line2_clip = TextClip(
                line2,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height//2)
            ).set_position(('center', main_title_y + title_fontsize//2 + 5))
            
            line1_clip = line1_clip.set_duration(video.duration)
            line2_clip = line2_clip.set_duration(video.duration)
            
            clips_to_combine.append(line1_clip)
            clips_to_combine.append(line2_clip)
            
            logger.info(f"Added multi-line title clips: '{line1}' and '{line2}'")
        else:
            # If no good break point, reduce font size
            while title_width_estimate > available_width and title_fontsize > 40:
                title_fontsize -= 5
                title_width_estimate = len(title_text) * title_fontsize * 0.6
            
            main_title_clip = TextClip(
                title_text,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height)
            ).set_position(('center', main_title_y))
            
            main_title_clip = main_title_clip.set_duration(video.duration)
            clips_to_combine.append(main_title_clip)
            logger.info(f"Added main title clip with reduced font size ({title_fontsize}px): '{title_text}'")
    else:
        # Title fits on one line
        main_title_clip = TextClip(
            title_text,
            fontsize=title_fontsize,
            color='#00FF00',
            font=font_path,
            size=(output_width - 2*title_margin, main_title_height)
        ).set_position(('center', main_title_y))
        
        main_title_clip = main_title_clip.set_duration(video.duration)
        clips_to_combine.append(main_title_clip)
        logger.info(f"Added main title clip: '{title_text}'")
    
    # Add service info
    service_info_clip = TextClip(
        "主日崇拜: 每週日下午2点",
        fontsize=40,
        color='white',
        font=font_path,
        size=(output_width - 2*title_margin, service_info_height)
    ).set_position(('center', service_info_y))
    service_info_clip = service_info_clip.set_duration(video.duration)
    clips_to_combine.append(service_info_clip)
    logger.info("Added service info clip")
    
    # Add subtitles only if requested and segments are available
    if add_subtitles and segments:
        # Filter segments that overlap with the highlight clip
        highlight_segments = [
            seg for seg in segments 
            if (seg.start <= highlight.end and seg.end >= highlight.start)
        ]
        logger.info(f"Found {len(highlight_segments)} subtitle segments in the selected time range")
        
        # Create a new list for processed segments
        processed_segments = []
        
        # Process word-level data from each segment
        for seg in highlight_segments:
            # Get words that fall within the highlight clip time range
            words = [
                word for word in seg.words
                if (word['start'] <= highlight.end and word['end'] >= highlight.start)
            ]
            
            if not words:
                continue
            
            # Group words into phrases (max 10 characters)
            current_phrase = []
            current_start = None
            current_text = ""
            
            for word in words:
                # Adjust timing relative to highlight start
                word_start = max(0, word['start'] - highlight.start)
                word_end = min(video.duration, word['end'] - highlight.start)
                
                # Initialize start time for new phrase
                if not current_start:
                    current_start = word_start
                
                # Add word to current phrase
                current_phrase.append(word)
                current_text += word['text']
                
                # Check if we should create a new phrase
                should_break = False
                
                # Break on punctuation, but only if we have enough content
                if any(p in word['text'] for p in '。，！？,.!?'):
                    # Only break if current phrase is long enough and not in the middle of a semantic unit
                    if len(current_text) >= 6:  # Lowered from 8
                        words_list = list(jieba.cut(current_text))
                        # Don't break if it would create a short next phrase
                        remaining_words = words[words.index(word)+1:]
                        if not remaining_words or len(''.join(w['text'] for w in remaining_words)) >= 6:  # Lowered from 8
                            should_break = True
                
                # Break on length
                elif len(current_text) >= 10:  # Lowered from 15
                    # Use jieba to find a natural break point
                    words_list = list(jieba.cut(current_text))
                    if len(words_list) > 1:
                        # Try to find the best break point near 10 characters
                        best_break_point = None
                        best_break_score = float('inf')
                        cumulative_length = 0
                        
                        for i, w in enumerate(words_list):
                            cumulative_length += len(w)
                            # Score based on how close to 10 characters and whether it's a natural break
                            score = abs(10 - cumulative_length)
                            # Prefer breaks at natural boundaries
                            if w in ['的', '了', '和', '與', '但是', '所以', '因為', '如果', '就是']:
                                score -= 2
                            
                            if cumulative_length >= 6 and cumulative_length <= 12:  # Allow 6-12 char range
                                if score < best_break_score:
                                    best_break_score = score
                                    best_break_point = cumulative_length
                        
                        if best_break_point:
                            should_break = True
                
                # Break on long pause (> 1.0 seconds), but only if we have enough content
                elif len(current_phrase) > 1:
                    pause_duration = word_start - (current_phrase[-2]['end'] - highlight.start)
                    if pause_duration > 1.0 and len(current_text) >= 6:  # Lowered from 8
                        # Check if breaking here would create a short next phrase
                        remaining_words = words[words.index(word)+1:]
                        if not remaining_words or len(''.join(w['text'] for w in remaining_words)) >= 6:  # Lowered from 8
                            should_break = True
                
                # Check for natural phrase breaks using jieba
                elif len(current_text) > 5:  # Only check if we have enough context
                    words_list = list(jieba.cut(current_text))
                    # Common Chinese particles and conjunctions that often mark phrase boundaries
                    phrase_boundaries = ['的', '了', '和', '與', '但是', '所以', '因為', '如果', '就是']  # Reduced list
                    
                    # Only break on boundaries if we have enough content
                    if 6 <= len(current_text) <= 12:  # Changed from >= 8 to a range
                        # Check if the last word is a natural boundary
                        if words_list[-1] in phrase_boundaries:
                            should_break = True
                        # Also break if we detect a complete semantic unit
                        elif len(words_list) >= 2 and any(w in phrase_boundaries for w in words_list[-2:]):
                            should_break = True

                if (should_break or word is words[-1]) and current_text:
                    # Convert text to traditional Chinese
                    phrase_text = zhconv.convert(current_text.strip(), 'zh-hant')
                    
                    # If this is the last word and the phrase is too short, try to combine with previous
                    if word is words[-1] and len(phrase_text) < 6 and processed_segments:  # Lowered from 5
                        prev_seg = processed_segments[-1]
                        combined_text = prev_seg.text + phrase_text
                        
                        # Only combine if the result isn't too long and the time gap isn't too large
                        if len(combined_text) <= 12:  # Lowered from 15
                            time_gap = current_start - prev_seg.end
                            if time_gap < 1.0:  # Less than 1 second gap
                                # Update previous segment
                                processed_segments[-1] = TranscriptionSegment(
                                    text=combined_text,
                                    start=prev_seg.start,
                                    end=word_end,
                                    words=prev_seg.words + current_phrase
                                )
                                # Reset without creating new segment
                                current_phrase = []
                                current_start = None
                                current_text = ""
                                continue
                    
                    # Don't create segments that are too short unless they're the only content
                    if len(phrase_text) >= 6 or word is words[-1]:  # Lowered from 5
                        new_seg = TranscriptionSegment(
                            text=phrase_text,
                            start=current_start,
                            end=word_end,
                            words=current_phrase.copy()
                        )
                        processed_segments.append(new_seg)
                    
                    # Reset for next phrase
                    current_phrase = []
                    current_start = None
                    current_text = ""
        
        # Sort segments by start time
        processed_segments.sort(key=lambda x: x.start)
        
        # Post-process segments to combine short ones
        i = 0
        while i < len(processed_segments) - 1:
            current_seg = processed_segments[i]
            next_seg = processed_segments[i + 1]
            
            # Check if either segment is too short or if combining would make a better length
            should_combine = False
            
            # If current segment is too short
            if len(current_seg.text) < 6:  # Lowered from 8
                should_combine = True
            # If next segment is too short
            elif len(next_seg.text) < 6:  # Lowered from 8
                should_combine = True
            # If both segments together would make a better phrase length (closer to 10)
            elif len(current_seg.text) + len(next_seg.text) <= 12:  # Lowered from 15
                time_gap = next_seg.start - current_seg.end
                if time_gap < 1.5:  # Allow slightly larger gap for better phrases
                    should_combine = True
            
            if should_combine:
                # Combine the segments
                combined_text = current_seg.text + next_seg.text
                if len(combined_text) <= 14:  # Lowered from 20
                    processed_segments[i] = TranscriptionSegment(
                        text=combined_text,
                        start=current_seg.start,
                        end=next_seg.end,
                        words=current_seg.words + next_seg.words
                    )
                    # Remove the next segment
                    processed_segments.pop(i + 1)
                    # Don't increment i, so we can check if the combined segment should be combined with the next
                    continue
            
            i += 1
        
        logger.info(f"After combining short segments: {len(processed_segments)} subtitle phrases")
        for i, seg in enumerate(processed_segments):
            logger.info(f"Segment {i+1}: {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
        
        # Calculate subtitle position right below the video
        subtitle_height = 120  # Increased height for larger text
        subtitle_fontsize = 65  # Increased font size
        subtitle_margin = 40
        
        # Position subtitle right below the video
        video_bottom = video_y + scaled_height
        subtitle_y = video_bottom + subtitle_margin
        
        logger.info(f"Positioning subtitles at y={subtitle_y} (below video)")
        logger.info(f"Using larger font size: {subtitle_fontsize}px")
        
        # Create subtitle clips for all processed segments
        for i, seg in enumerate(processed_segments):
            logger.info(f"Creating subtitle {i+1}/{len(processed_segments)}: {seg.start:.2f}s - {seg.end:.2f}s")
            logger.info(f"Text: {seg.text}")
            
            # Create subtitle with animation
            subtitle = TextClip(
                seg.text,
                fontsize=subtitle_fontsize,
                color='white',
                font=font_path,
                method='caption',
                align='center',
                size=(output_width - 2*subtitle_margin, subtitle_height)
            )
            
            # Add fade in/out animation
            fade_duration = min(0.3, (seg.end - seg.start) / 4)
            subtitle = subtitle.set_position(('center', subtitle_y))
            subtitle = subtitle.set_start(seg.start).set_end(seg.end)
            subtitle = subtitle.crossfadein(fade_duration).crossfadeout(fade_duration)
            
            clips_to_combine.append(subtitle)
    else:
        if add_subtitles and not segments:
            logger.info("Skipping subtitles as no segments are available (full video mode)")
        else:
            logger.info("Skipping subtitles as they were not requested")
    
    # Combine all clips
    logger.info(f"Combining {len(clips_to_combine)} clips")
    final_clip = CompositeVideoClip(clips_to_combine, size=(output_width, output_height))
    final_clip = final_clip.set_duration(video.duration)
    
    # Write output file
    logger.info(f"Writing output video to {output_path}")
    logger.info(f"Video duration: {video.duration:.2f}s")
    logger.info(f"Output dimensions: {output_width}x{output_height}")
    logger.info(f"FPS: {video.fps}")
    
    final_clip.write_videofile(
        output_path,
        fps=video.fps,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    logger.info("Highlight video creation complete")

def process_single_file(
    input_file: str,
    output_file: Optional[str],
    args: argparse.Namespace
) -> bool:
    """
    Process a single file with the given arguments.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file (optional)
        args: Command line arguments
        
    Returns:
        bool: True if processing was successful
    """
    try:
        # Check if input is audio or video
        is_video = input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        
        # If audio file and not transcribe-only, skip
        if not is_video and not args.transcribe_only:
            logger.warning(f"Skipping audio file {input_file} (use --transcribe-only for audio files)")
            return False
        
        # Default output path for video files
        if not output_file and is_video:
            input_path = Path(input_file)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(input_path.with_name(f"{input_path.stem}_highlight_{timestamp}{input_path.suffix}"))
            logger.info(f"Setting default output path to: {output_file}")

        # Step 1: Load or generate transcription
        if args.resume_from in ['transcribe', 'segments', 'titles']:
            # Load transcription state
            transcript_data = load_state(input_file, 'transcript')
            if transcript_data:
                segments = [
                    TranscriptionSegment(
                        text=seg['text'],
                        start=seg['start'],
                        end=seg['end'],
                        words=seg.get('words', []),
                        speaker=seg.get('speaker')
                    )
                    for seg in transcript_data
                ]
                logger.info(f"Loaded transcription with {len(segments)} segments")
                
                # If using transcribe_only mode, create text transcript file
                if args.transcribe_only:
                    transcript_txt_path = Path(input_file).with_suffix('.transcript.txt')
                    if not transcript_txt_path.exists():
                        logger.info("Creating plain text transcript file from loaded state...")
                        create_text_transcript(input_file, segments)
                    logger.info("Transcription data loaded. Exiting as --transcribe-only flag was set.")
                    return True
            else:
                logger.error("No transcription state found to resume from")
                if args.full_video and is_video:
                    logger.info("Continuing without transcription in full video mode")
                    segments = None
                else:
                    return False
        else:
            logger.info("Generating transcription...")
            segments = transcribe_video(input_file, args.language_code)
            
            # If transcribe_only flag is set, exit after transcription
            if args.transcribe_only:
                # Create plain text transcript file if it doesn't exist
                transcript_txt_path = Path(input_file).with_suffix('.transcript.txt')
                if not transcript_txt_path.exists():
                    logger.info("Creating plain text transcript file...")
                    create_text_transcript(input_file, segments)
                
                logger.info("Transcription completed.")
                return True

        # Skip video processing for audio files
        if not is_video:
            return True
        
        # Step 2: Load or identify highlight segments
        if args.full_video:
            logger.info("Using full video mode (skipping segment identification)")
            segments_data = create_full_video_segment(input_file)
        elif args.resume_from in ['segments', 'titles']:
            segments_data = load_state(input_file, 'segments')
            if not segments_data:
                logger.error("No segments state found to resume from")
                return False
        else:
            logger.info("Identifying engaging segments...")
            segments_data = identify_highlights(segments, args.max_duration)
            save_state(input_file, 'segments', segments_data)

        # Step 3: Generate or load titles
        if args.manual_title:
            main_title = args.manual_title
            highlights = []
            for seg in segments_data['segments']:
                highlight = HighlightClip(
                    start=float(seg['start_time']),
                    end=float(seg['end_time']),
                    score=80.0,  # Default score
                    title=main_title
                )
                highlights.append(highlight)
        elif args.resume_from == 'titles':
            titles_data = load_state(input_file, 'titles')
            if titles_data:
                main_title = titles_data.get('main_title', '')
                highlights = [
                    HighlightClip(
                        start=float(h['start']),
                        end=float(h['end']),
                        score=float(h['score']),
                        title=h['title']
                    )
                    for h in titles_data.get('highlights', [])
                ]
            else:
                logger.error("No titles state found to resume from")
                return False
        else:
            logger.info("Generating titles...")
            main_title, highlights = generate_titles(segments_data)
            titles_data = {
                'main_title': main_title,
                'highlights': [
                    {
                        'start': h.start,
                        'end': h.end,
                        'score': h.score,
                        'title': h.title
                    }
                    for h in highlights
                ]
            }
            save_state(input_file, 'titles', titles_data)

        # Step 4: Create highlight video
        if not highlights:
            logger.error("No highlights found to process")
            return False

        highlight = highlights[0]  # Use the first (or only) highlight
        logger.info(f"Selected highlight: {highlight}")

        # Create the highlight video
        create_highlight_video(
            input_path=input_file,
            highlight=highlight,
            segments=segments,
            output_path=output_file,
            main_title=main_title or highlight.title,
            logo_path=args.logo,
            add_subtitles=args.add_subtitles,
            vertical_position_ratio=args.vertical_position,
            background_image_path=args.background_image,
            zoom_factor=args.zoom_factor
        )
        
        logger.info(f"Highlight video created successfully: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        return False

def main():
    """Main function to process video and create highlight clips"""
    parser = argparse.ArgumentParser(description='Generate highlight clips from long videos')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', '-i', help='Path to the input video or audio file')
    input_group.add_argument('--input-folder', '-f', help='Path to folder containing video/audio files')
    parser.add_argument('--output', '-o', help='Path to the output video file (only used with --input-file)')
    parser.add_argument('--max-duration', '-d', type=float, default=120.0, 
                      help='Target duration for the highlight clip in seconds (default: 120)')
    parser.add_argument('--logo', '-l', 
                      default=os.path.join(os.path.dirname(__file__), 'default_assets', 'Waymaker_white_logo_transparent_background.png'),
                      help='Path to logo image file (default: default_assets/Waymaker_white_logo_transparent_background.png)')
    parser.add_argument('--background-image', '-b', 
                      help='Path to banner image to display at the top of the video (default: none)')
    parser.add_argument('--zoom-factor', '-z', type=float, default=2.0,
                      help='Zoom factor for the video (default: 2.0 for 200%% zoom)')
    parser.add_argument('--resume-from', '-r', choices=['transcribe', 'segments', 'titles'],
                      help='Resume from a specific stage using saved state')
    parser.add_argument('--add-subtitles', action='store_true',
                      help='Add subtitles to the video (default: False)')
    parser.add_argument('--vertical-position', '-v', type=float, default=0.67,
                      help='Vertical position ratio for the main subject (default: 0.67 for 2/3 from top)')
    parser.add_argument('--full-video', '-F', action='store_true',
                      help='Process the entire video without selecting segments (default: False)')
    parser.add_argument('--manual-title', '-m', help='Manually specify the video title')
    parser.add_argument('--language-code', type=str, default='zh',
                      help='Language code for transcription (default: zh for Chinese)')
    parser.add_argument('--transcribe-only', action='store_true',
                      help='Stop after transcription step and exit')
    
    args = parser.parse_args()
    
    # Ensure default logo directory exists
    default_logo_dir = os.path.join(os.path.dirname(__file__), 'default_assets')
    if not os.path.exists(default_logo_dir):
        os.makedirs(default_logo_dir)
        logger.info(f"Created default assets directory: {default_logo_dir}")
    
    # Check if logo file exists
    if not os.path.exists(args.logo):
        logger.warning(f"Logo file not found at {args.logo}")
        args.logo = None  # Fall back to text if logo file doesn't exist
    
    try:
        if args.input_folder:
            # Process folder
            files = find_supported_files(args.input_folder)
            if not files:
                logger.error("No supported files found in the specified folder")
                return
            
            # Process each file
            total_files = len(files)
            successful = 0
            failed = 0
            
            for i, input_file in enumerate(files, 1):
                logger.info(f"\nProcessing file {i}/{total_files}: {input_file}")
                
                # Process the file (output path will be generated in process_single_file)
                if process_single_file(input_file, None, args):
                    successful += 1
                else:
                    failed += 1
            
            # Print summary
            logger.info("\nProcessing Summary:")
            logger.info(f"Total files: {total_files}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
        
        else:
            # Process single file
            process_single_file(args.input_file, args.output, args)
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 