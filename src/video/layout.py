#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video layout module for compositing and arranging video elements.
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import (
    TextClip, ImageClip, CompositeVideoClip, 
    clips_array, vfx, concatenate_videoclips
)

logger = logging.getLogger(__name__)

def create_portrait_layout(main_clip: VideoFileClip, 
                          target_resolution: Tuple[int, int] = (1080, 1920),
                          background_color: Tuple[int, int, int] = (0, 0, 0)) -> CompositeVideoClip:
    """
    Create a portrait layout for the main video clip, suitable for platforms like TikTok, Instagram Reels, etc.
    
    Args:
        main_clip: The main video clip to display
        target_resolution: Output resolution (width, height)
        background_color: RGB background color
        
    Returns:
        CompositeVideoClip with the portrait layout
    """
    target_width, target_height = target_resolution
    
    # Calculate the aspect ratio of the main clip
    main_aspect = main_clip.w / main_clip.h
    target_aspect = target_width / target_height
    
    # Determine if we need to fit by width or height
    if main_aspect > target_aspect:
        # Wider than target, fit to width
        new_width = target_width
        new_height = int(new_width / main_aspect)
    else:
        # Taller than target, fit to height
        new_height = target_height
        new_width = int(new_height * main_aspect)
    
    # Resize the main clip
    resized_clip = main_clip.resize((new_width, new_height))
    
    # Create a background clip
    bg_clip = ImageClip(
        np.ones((target_height, target_width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8),
        duration=main_clip.duration
    )
    
    # Position the resized clip in the center
    x_pos = (target_width - new_width) // 2
    y_pos = (target_height - new_height) // 2
    
    # Compose the clips
    composite = CompositeVideoClip(
        [bg_clip, resized_clip.set_position((x_pos, y_pos))],
        size=target_resolution
    )
    
    # Preserve audio
    if main_clip.audio is not None:
        composite = composite.set_audio(main_clip.audio)
    
    return composite

def add_caption_to_video(clip: VideoFileClip, 
                        text: str, 
                        font_size: int = 40,
                        color: str = 'white',
                        bg_color: Optional[str] = 'black',
                        position: Tuple[str, str] = ('center', 'bottom'),
                        padding: int = 20) -> CompositeVideoClip:
    """
    Add a caption to a video clip.
    
    Args:
        clip: The video clip to add a caption to
        text: The text to display
        font_size: Font size
        color: Text color
        bg_color: Background color (None for transparent)
        position: Position as ('horizontal', 'vertical') where each can be 'left'/'center'/'right' or 'top'/'center'/'bottom'
        padding: Padding around the text
        
    Returns:
        CompositeVideoClip with the caption added
    """
    # Create the text clip
    txt_clip = TextClip(
        text, 
        fontsize=font_size, 
        color=color,
        bg_color=bg_color,
        method='caption',
        align='center',
        stroke_color='black',
        stroke_width=2,
        size=(clip.w - padding * 2, None)
    )
    
    # Set the duration to match the video
    txt_clip = txt_clip.set_duration(clip.duration)
    
    # Determine position
    h_pos, v_pos = position
    
    if h_pos == 'left':
        x_pos = padding
    elif h_pos == 'center':
        x_pos = 'center'
    else:  # right
        x_pos = clip.w - txt_clip.w - padding
    
    if v_pos == 'top':
        y_pos = padding
    elif v_pos == 'center':
        y_pos = 'center'
    else:  # bottom
        y_pos = clip.h - txt_clip.h - padding
    
    # Compose the clips
    composite = CompositeVideoClip([
        clip,
        txt_clip.set_position((x_pos, y_pos))
    ], size=clip.size)
    
    # Preserve audio
    if clip.audio is not None:
        composite = composite.set_audio(clip.audio)
    
    return composite

def create_subtitle_clip(text: str,
                        duration: float,
                        font_size: int = 30,
                        color: str = 'white',
                        bg_color: Optional[str] = 'black',
                        width: int = 1000) -> TextClip:
    """
    Create a subtitle text clip.
    
    Args:
        text: The text to display
        duration: Duration in seconds
        font_size: Font size
        color: Text color
        bg_color: Background color (None for transparent)
        width: Maximum width of the text
        
    Returns:
        TextClip object
    """
    # Create the text clip
    txt_clip = TextClip(
        text, 
        fontsize=font_size, 
        color=color,
        bg_color=bg_color,
        method='caption',
        align='center',
        stroke_color='black',
        stroke_width=2,
        size=(width, None)
    )
    
    # Set the duration
    txt_clip = txt_clip.set_duration(duration)
    
    return txt_clip

def add_subtitles_to_video(clip: VideoFileClip, 
                          subtitles: List[Dict[str, Any]],
                          font_size: int = 30,
                          position: Tuple[str, str] = ('center', 'bottom'),
                          padding: int = 20) -> CompositeVideoClip:
    """
    Add subtitles to a video clip.
    
    Args:
        clip: The video clip to add subtitles to
        subtitles: List of subtitle dicts with 'text', 'start', 'end' keys
        font_size: Font size
        position: Position as ('horizontal', 'vertical')
        padding: Padding around the text
        
    Returns:
        CompositeVideoClip with subtitles added
    """
    # Create a list to hold all text clips
    txt_clips = []
    
    # Process each subtitle
    for sub in subtitles:
        # Extract subtitle info
        text = sub['text']
        start_time = sub['start']
        end_time = sub['end']
        duration = end_time - start_time
        
        # Create a text clip for this subtitle
        txt_clip = create_subtitle_clip(
            text,
            duration=duration,
            font_size=font_size,
            width=clip.w - padding * 2
        )
        
        # Set the position and start time
        h_pos, v_pos = position
        
        if h_pos == 'left':
            x_pos = padding
        elif h_pos == 'center':
            x_pos = 'center'
        else:  # right
            x_pos = clip.w - txt_clip.w - padding
        
        if v_pos == 'top':
            y_pos = padding
        elif v_pos == 'center':
            y_pos = 'center'
        else:  # bottom
            y_pos = clip.h - txt_clip.h - padding
        
        # Set position and start time
        txt_clip = txt_clip.set_position((x_pos, y_pos)).set_start(start_time)
        
        # Add to the list
        txt_clips.append(txt_clip)
    
    # Create the composite
    composite = CompositeVideoClip([clip] + txt_clips, size=clip.size)
    
    # Preserve audio
    if clip.audio is not None:
        composite = composite.set_audio(clip.audio)
    
    return composite

def create_highlight_compilation(clips: List[VideoFileClip], 
                               fade_duration: float = 0.5,
                               target_resolution: Optional[Tuple[int, int]] = None) -> VideoFileClip:
    """
    Create a compilation of highlight clips with smooth transitions.
    
    Args:
        clips: List of video clips to compile
        fade_duration: Duration of crossfade between clips in seconds
        target_resolution: Target resolution for all clips (width, height) or None to keep original
        
    Returns:
        Compiled VideoFileClip
    """
    # Make sure we have clips to work with
    if not clips:
        logger.error("No clips provided for compilation")
        raise ValueError("No clips provided for compilation")
    
    # Resize clips if target resolution is specified
    if target_resolution:
        resized_clips = []
        for clip in clips:
            resized_clips.append(clip.resize(target_resolution))
        clips = resized_clips
    
    # Apply crossfade effect
    if len(clips) > 1 and fade_duration > 0:
        final_clip = concatenate_videoclips(
            clips, 
            method="compose",
            padding=-fade_duration,  # Negative padding creates crossfade
            transition=lambda c1, c2: CompositeVideoClip([
                c1.crossfadeout(fade_duration),
                c2.crossfadein(fade_duration)
            ])
        )
    else:
        # Just concatenate if only one clip or no fade
        final_clip = concatenate_videoclips(clips, method="compose")
    
    return final_clip

def create_split_screen(clips: List[VideoFileClip], 
                      layout: str = 'horizontal',
                      background_color: Tuple[int, int, int] = (0, 0, 0)) -> CompositeVideoClip:
    """
    Create a split screen layout with multiple clips.
    
    Args:
        clips: List of video clips to arrange
        layout: Layout type ('horizontal', 'vertical', 'grid')
        background_color: RGB background color
        
    Returns:
        CompositeVideoClip with the split screen layout
    """
    # Determine the layout
    if layout == 'horizontal':
        # Calculate the common height
        heights = [clip.h for clip in clips]
        common_height = min(heights)
        
        # Resize clips to have the same height
        resized_clips = []
        for clip in clips:
            new_width = int(clip.w * (common_height / clip.h))
            resized_clips.append(clip.resize(height=common_height))
        
        # Arrange clips horizontally
        arranged_clip = clips_array([resized_clips])
        
    elif layout == 'vertical':
        # Calculate the common width
        widths = [clip.w for clip in clips]
        common_width = min(widths)
        
        # Resize clips to have the same width
        resized_clips = []
        for clip in clips:
            new_height = int(clip.h * (common_width / clip.w))
            resized_clips.append(clip.resize(width=common_width))
        
        # Convert to a column
        clips_column = [[clip] for clip in resized_clips]
        
        # Arrange clips vertically
        arranged_clip = clips_array(clips_column)
        
    elif layout == 'grid':
        # Calculate grid dimensions
        n_clips = len(clips)
        cols = int(np.ceil(np.sqrt(n_clips)))
        rows = int(np.ceil(n_clips / cols))
        
        # Pad with None if needed
        padded_clips = clips + [None] * (rows * cols - n_clips)
        
        # Convert to a grid
        clips_grid = []
        for i in range(rows):
            row = padded_clips[i * cols: (i + 1) * cols]
            clips_grid.append(row)
        
        # Arrange clips in a grid
        arranged_clip = clips_array(clips_grid)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    # Determine which clip has audio
    for clip in clips:
        if clip.audio is not None:
            arranged_clip = arranged_clip.set_audio(clip.audio)
            break
    
    return arranged_clip

def add_progress_bar(clip: VideoFileClip, 
                   height: int = 5,
                   color: Tuple[int, int, int] = (255, 0, 0),
                   position: str = 'bottom') -> VideoFileClip:
    """
    Add a progress bar to a video clip.
    
    Args:
        clip: The video clip to add a progress bar to
        height: Height of the progress bar in pixels
        color: RGB color of the progress bar
        position: Position of the bar ('top' or 'bottom')
        
    Returns:
        VideoFileClip with a progress bar
    """
    def make_frame(t):
        # Get the current frame
        frame = clip.get_frame(t)
        
        # Calculate progress as a fraction (0 to 1)
        progress = t / clip.duration
        
        # Create a copy of the frame to modify
        result = frame.copy()
        
        # Calculate the width of the progress bar
        bar_width = int(clip.w * progress)
        
        # Determine position
        if position == 'top':
            y_start = 0
        else:  # bottom
            y_start = clip.h - height
        
        # Draw the progress bar
        result[y_start:y_start + height, 0:bar_width] = color
        
        return result
    
    # Create a new clip with the progress bar
    result_clip = VideoFileClip(make_frame=make_frame, duration=clip.duration)
    
    # Ensure we keep the original audio
    if clip.audio is not None:
        result_clip = result_clip.set_audio(clip.audio)
    
    return result_clip 