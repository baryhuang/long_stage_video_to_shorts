#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layout preview module for generating preview images of video layouts.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.models.font import FontManager
from src.models.transcription import TranscriptionSegment
from src.models.highlight import HighlightClip
from src.video.processor import track_and_zoom_video

logger = logging.getLogger(__name__)

def create_layout_preview(
    input_path: str,
    timestamp: float,
    main_title: str,
    output_preview_path: str,
    logo_path: Optional[str] = None,
    segments: Optional[List[TranscriptionSegment]] = None,
    highlight: Optional[HighlightClip] = None,
    zoom_factor: float = 2.0,
    vertical_position_ratio: float = 0.67,
    skip_logo: bool = False,
    skip_background: bool = False,
    skip_service_info: bool = False,
    background_image_path: Optional[str] = None
) -> bool:
    """
    Create a preview image of the video layout
    
    Args:
        input_path: Path to input video
        timestamp: Timestamp to use for preview frame
        main_title: Main title of the video
        output_preview_path: Path to save the preview image
        logo_path: Path to logo image file (optional)
        segments: List of transcription segments (optional)
        highlight: HighlightClip object (optional)
        zoom_factor: Factor to zoom the video (default: 2.0)
        vertical_position_ratio: Ratio for vertical positioning (default: 0.67)
        skip_logo: Skip logo/church name display (default: False)
        skip_background: Skip background/banner image display (default: False)
        background_image_path: Path to background image file (optional)
    
    Returns:
        bool: True if user confirms the layout, False otherwise
    """
    logger.info(f"Creating layout preview using frame at {timestamp:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    logger.info(f"Using font: {font_path}")
    
    # Load video and extract frame
    video = VideoFileClip(input_path)
    frame = video.get_frame(timestamp)
    video.close()
    
    # Debug original video dimensions
    logger.info(f"Original video dimensions: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    
    # Create a temporary clip for the single frame with zoom and tracking
    frame_clip = VideoFileClip(input_path).subclip(timestamp, timestamp + 0.1)
    processed_clip = track_and_zoom_video(
        frame_clip,
        min_zoom=zoom_factor,
        max_zoom=zoom_factor,
        smoothing_window=15,
        vertical_position_ratio=vertical_position_ratio
    )
    processed_frame = processed_clip.get_frame(0)
    processed_clip.close()
    frame_clip.close()
    
    # Create PIL Image for preview
    frame_image = Image.fromarray(processed_frame)
    
    # Calculate dimensions for 9:16 format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Set video section to exactly 50% of total height
    video_section_height = int(output_height * 0.5)  # 50% of total height
    
    # Calculate the width while maintaining aspect ratio
    aspect_ratio = frame_image.width / frame_image.height
    video_section_width = int(video_section_height * aspect_ratio)
    
    # Resize frame to fit the calculated dimensions (always use 50% height, allow width to exceed screen)
    frame_resized = frame_image.resize((video_section_width, video_section_height), Image.Resampling.LANCZOS)
    logger.info(f"Resized video frame dimensions: {video_section_width}x{video_section_height} (width x height)")
    
    # Calculate position to center the frame exactly horizontally
    frame_x = (output_width - video_section_width) // 2
    video_section_y = (output_height - video_section_height) // 2  # Center vertically
    logger.info(f"Video frame position: x={frame_x}, y={video_section_y}")
    
    # Create a temporary image that can hold the oversized frame if needed
    temp_width = max(output_width, video_section_width)
    temp_preview = Image.new('RGB', (temp_width, output_height), 'black')
    
    # Paste the frame at the calculated position
    temp_preview.paste(frame_resized, (frame_x, video_section_y))
    
    # Crop the center portion if the width exceeds the output width
    if temp_width > output_width:
        left = (temp_width - output_width) // 2
        preview = temp_preview.crop((left, 0, left + output_width, output_height))
    else:
        preview = temp_preview
    
    # Create draw object
    draw = ImageDraw.Draw(preview)
    
    # Load fonts with proper Chinese support
    title_font = ImageFont.truetype(font_path, 80)
    church_font = ImageFont.truetype(font_path, 50)
    service_font = ImageFont.truetype(font_path, 40)
    
    # Define text space heights (reduced for more video space)
    title_margin = 25
    church_name_height = 100
    main_title_height = 100
    service_info_height = 50
    
    # Calculate text positions above the video section
    # Position text elements above the video
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
    
    # Position subtitle below video section
    subtitle_margin = 40
    subtitle_y = video_section_y + video_section_height + subtitle_margin
    
    logger.info(f"Text element positions:")
    logger.info(f"- Church name: y={church_name_y}")
    logger.info(f"- Main title: y={main_title_y}")
    logger.info(f"- Service info: y={service_info_y}")
    logger.info(f"- Subtitle position: y={subtitle_y}")
    
    # Add logo for church name
    if not skip_logo:
        if logo_path and os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path)
                # Calculate logo size to fit church name height while maintaining aspect ratio
                logo_height = church_name_height
                logo_width = int(logo.width * (logo_height / logo.height))
                
                # Allow logo to be wider, up to 90% of output width
                max_logo_width = int(output_width * 0.9)
                if logo_width > max_logo_width:
                    logo_width = max_logo_width
                    logo_height = int(logo.height * (logo_width / logo.width))
                
                logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                logger.info(f"Adding logo with dimensions: {logo_width}x{logo_height}")
                
                # Calculate position to center the logo
                logo_x = (output_width - logo_width) // 2
                logger.info(f"Logo position: x={logo_x}, y={church_name_y}")
                
                # Paste logo with alpha channel support
                preview.paste(logo_resized, (logo_x, church_name_y), logo_resized if logo.mode == 'RGBA' else None)
            except Exception as e:
                logger.warning(f"Failed to add logo as church name: {e}")
                # Fallback to text if logo fails
                church_name = "開路者教會 WAYMAKER CHURCH"
                logger.info(f"Falling back to text: '{church_name}' at y={church_name_y}")
                draw.text(
                    (output_width//2, church_name_y + church_name_height//2),
                    church_name,
                    font=church_font,
                    fill='white',
                    anchor='mm'
                )
        else:
            # Fallback to text if no logo provided
            church_name = "開路者教會 WAYMAKER CHURCH"
            logger.info(f"No logo provided, using text: '{church_name}' at y={church_name_y}")
            draw.text(
                (output_width//2, church_name_y + church_name_height//2),
                church_name,
                font=church_font,
                fill='white',
                anchor='mm'
            )
    else:
        logger.info("Skipping logo/church name as requested")
    
    # Draw main title with multi-line support
    logger.info(f"Drawing main title: '{main_title}' at y={main_title_y}")
    
    # Calculate text size and wrap if needed
    title_font_size = 80
    title_font = ImageFont.truetype(font_path, title_font_size)
    
    # Check if title fits within width
    title_width = title_font.getbbox(main_title)[2] - title_font.getbbox(main_title)[0]
    available_width = output_width - 2 * title_margin
    
    if title_width > available_width:
        # Try to find a natural break point
        break_points = []
        for i, char in enumerate(main_title):
            if char in '。，！？,.!? ':
                break_points.append(i)
        
        if break_points and len(main_title) > 10:
            # Find the best break point near the middle
            middle_idx = len(main_title) // 2
            best_break = min(break_points, key=lambda x: abs(x - middle_idx))
            
            # Split the title into two lines
            line1 = main_title[:best_break+1]
            line2 = main_title[best_break+1:]
            
            # Draw each line
            line_spacing = 10
            draw.text(
                (output_width//2, main_title_y - title_font_size//2 - line_spacing//2),
                line1,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
            draw.text(
                (output_width//2, main_title_y + title_font_size//2 + line_spacing//2),
                line2,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
        else:
            # If no good break point, reduce font size
            while title_width > available_width and title_font_size > 40:
                title_font_size -= 5
                title_font = ImageFont.truetype(font_path, title_font_size)
                title_width = title_font.getbbox(main_title)[2] - title_font.getbbox(main_title)[0]
            
            draw.text(
                (output_width//2, main_title_y),
                main_title,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
    else:
        # Title fits, draw normally
        draw.text(
            (output_width//2, main_title_y),
            main_title,
            font=title_font,
            fill='#00FF00',
            anchor='mm'
        )
    
    # Draw service info (if not skipped)
    if not skip_service_info:
        service_info = "主日崇拜: 每週日下午2点"
        logger.info(f"Drawing service info: '{service_info}' at y={service_info_y}")
        draw.text(
            (output_width//2, service_info_y),
            service_info,
            font=service_font,
            fill='white',
            anchor='mm'
        )
    else:
        logger.info("Skipped service info in preview")
    
    # Add first subtitle if segments are provided
    if segments and highlight:
        # Find the first segment that overlaps with the highlight
        first_subtitle = None
        for seg in segments:
            if seg.start <= highlight.end and seg.end >= highlight.start:
                first_subtitle = seg
                break
        
        if first_subtitle:
            # Split text if longer than 15 characters
            subtitle_text = first_subtitle.text
            if len(subtitle_text) > 15:
                # Try to find a natural break point
                break_points = [i for i, char in enumerate(subtitle_text[:15]) if char in '。，！？,.!? ']
                if break_points:
                    subtitle_text = subtitle_text[:max(break_points) + 1]
                else:
                    subtitle_text = subtitle_text[:15]
            
            # Calculate subtitle position below video
            subtitle_height = 120
            subtitle_margin = 40
            subtitle_y = video_section_y + video_section_height + subtitle_margin
            
            # Create subtitle font
            subtitle_font = ImageFont.truetype(font_path, 65)
            
            logger.info(f"Adding preview subtitle: '{subtitle_text}' at y={subtitle_y}")
            draw.text(
                (output_width//2, subtitle_y + subtitle_height//2),
                subtitle_text,
                font=subtitle_font,
                fill='white',
                anchor='mm'
            )
    
    # Save preview
    preview.save(output_preview_path, 'JPEG', quality=95)
    logger.info(f"Saved layout preview to {output_preview_path}")
    
    # Show preview and get confirmation
    print("\nLayout Preview:")
    print("-" * 50)
    print("Layout dimensions:")
    print(f"- Output size: {output_width}x{output_height}")
    print(f"- Video frame: {video_section_width}x{video_section_height} at y={video_section_y}")
    print(f"- Text margins: {title_margin}px")
    print("\nText positions (y-coordinate from top):")
    print(f"- Church name area: {church_name_y}px")
    print(f"- Main title: {main_title_y}px")
    print(f"- Service info: {service_info_y}px")
    print(f"\nPreview image saved to: {output_preview_path}")
    print("\nPlease check the preview image and confirm if you want to proceed with the video creation.")
    
    while True:
        choice = input("\nDo you want to:\n1. Continue with this layout\n2. Exit\nEnter your choice (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    return choice == '1' 