#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video processing module for handling video editing, effects, and transformations.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, clips_array
from moviepy.video.fx import all as vfx
import cv2

from .transcription import Transcription, TranscriptionSegment

logger = logging.getLogger(__name__)

def load_video(video_path: Union[str, Path]) -> VideoFileClip:
    """
    Load a video file into a VideoFileClip object.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        MoviePy VideoFileClip object
    """
    logger.info(f"Loading video: {video_path}")
    return VideoFileClip(str(video_path))

def save_video(clip: VideoFileClip, 
              output_path: Union[str, Path], 
              codec: str = 'libx264', 
              bitrate: str = '8000k',
              audio_codec: str = 'aac',
              audio_bitrate: str = '192k',
              fps: Optional[float] = None,
              preset: str = 'medium',
              threads: int = 2,
              remove_temp: bool = True) -> str:
    """
    Save a video clip to file with the specified settings.
    
    Args:
        clip: MoviePy VideoFileClip object
        output_path: Path to save the video file
        codec: Video codec to use
        bitrate: Video bitrate
        audio_codec: Audio codec to use
        audio_bitrate: Audio bitrate
        fps: Frames per second (if None, uses original FPS)
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        threads: Number of threads to use for encoding
        remove_temp: Whether to remove temporary files after encoding
        
    Returns:
        Path to the saved video file
    """
    logger.info(f"Saving video to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # If fps is not specified, use the original fps
    if fps is None and hasattr(clip, 'fps'):
        fps = clip.fps
    
    # Save the video
    clip.write_videofile(
        str(output_path),
        codec=codec,
        bitrate=bitrate,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        fps=fps,
        preset=preset,
        threads=threads,
        remove_temp=remove_temp
    )
    
    logger.info(f"Video saved to: {output_path}")
    return str(output_path)

def extract_clip(video: VideoFileClip, 
               start_time: float, 
               end_time: float,
               resize_height: Optional[int] = None) -> VideoFileClip:
    """
    Extract a clip from a video between specified start and end times.
    
    Args:
        video: MoviePy VideoFileClip object
        start_time: Start time in seconds
        end_time: End time in seconds
        resize_height: Optional height to resize the clip to (maintains aspect ratio)
        
    Returns:
        Extracted clip as a VideoFileClip
    """
    # Ensure times are within bounds
    start_time = max(0, start_time)
    end_time = min(video.duration, end_time)
    
    if start_time >= end_time:
        raise ValueError(f"Invalid time range: start_time ({start_time}) must be less than end_time ({end_time})")
    
    # Extract the subclip
    subclip = video.subclip(start_time, end_time)
    
    # Resize if requested
    if resize_height is not None:
        subclip = subclip.resize(height=resize_height)
    
    return subclip

def apply_zoom_effect(clip: VideoFileClip, 
                    zoom_factor: float = 1.2,
                    center: Optional[Tuple[float, float]] = None) -> VideoFileClip:
    """
    Apply a zoom effect to a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        zoom_factor: How much to zoom in (1.0 = no zoom)
        center: (x, y) center point for zoom as fractions of width and height (0-1)
                If None, zooms into the center of the frame
        
    Returns:
        Zoomed clip
    """
    # Default to center if not specified
    if center is None:
        center = (0.5, 0.5)
    
    # Create zoom effect
    def zoom_effect(frame, t):
        # Get dimensions of the frame
        h, w = frame.shape[:2]
        
        # Calculate the center point in pixels
        center_x = int(w * center[0])
        center_y = int(h * center[1])
        
        # Calculate zoom matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        
        # Apply the zoom
        zoomed_frame = cv2.warpAffine(frame, M, (w, h))
        return zoomed_frame
    
    # Apply the effect
    zoomed_clip = clip.fl_image(zoom_effect)
    return zoomed_clip

def apply_pan_effect(clip: VideoFileClip, 
                   start_pos: Tuple[float, float] = (0.5, 0.5),
                   end_pos: Tuple[float, float] = (0.5, 0.5),
                   zoom_factor: float = 1.2) -> VideoFileClip:
    """
    Apply a pan effect to a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        start_pos: (x, y) starting position as fractions of width and height (0-1)
        end_pos: (x, y) ending position as fractions of width and height (0-1)
        zoom_factor: Base zoom factor to apply
        
    Returns:
        Panned clip
    """
    def pan_effect(frame, t):
        # Get dimensions of the frame
        h, w = frame.shape[:2]
        
        # Calculate the interpolated center point at this time
        progress = t / clip.duration
        center_x = int(w * (start_pos[0] + (end_pos[0] - start_pos[0]) * progress))
        center_y = int(h * (start_pos[1] + (end_pos[1] - start_pos[1]) * progress))
        
        # Calculate zoom matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)
        
        # Apply the zoom and pan
        panned_frame = cv2.warpAffine(frame, M, (w, h))
        return panned_frame
    
    # Apply the effect
    panned_clip = clip.fl_image(pan_effect)
    return panned_clip

def apply_fade_in_out(clip: VideoFileClip, 
                    fade_in_duration: float = 0.5,
                    fade_out_duration: float = 0.5) -> VideoFileClip:
    """
    Apply fade in and fade out effects to a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        fade_in_duration: Duration of fade in effect in seconds
        fade_out_duration: Duration of fade out effect in seconds
        
    Returns:
        Clip with fade effects applied
    """
    # Apply fade effects
    clip = clip.fadein(fade_in_duration)
    clip = clip.fadeout(fade_out_duration)
    return clip

def create_grid_composition(clips: List[VideoFileClip], 
                          rows: int, 
                          cols: int) -> CompositeVideoClip:
    """
    Create a grid composition of video clips.
    
    Args:
        clips: List of MoviePy VideoFileClip objects
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        CompositeVideoClip with clips arranged in a grid
    """
    # Ensure we have enough clips for the grid
    if len(clips) < rows * cols:
        # Fill with black clips if needed
        for _ in range(rows * cols - len(clips)):
            black_clip = VideoFileClip(None, duration=clips[0].duration)
            black_clip = black_clip.set_opacity(0)
            clips.append(black_clip)
    
    # Trim excess clips
    clips = clips[:rows * cols]
    
    # Reshape the clips list into a 2D array
    grid = []
    for i in range(rows):
        row = clips[i * cols:(i + 1) * cols]
        grid.append(row)
    
    # Create the grid composition
    composition = clips_array(grid)
    return composition

def detect_scenes(video_path: Union[str, Path], 
                threshold: float = 30.0, 
                min_scene_duration: float = 2.0) -> List[Tuple[float, float]]:
    """
    Detect scene changes in a video.
    
    Args:
        video_path: Path to the video file
        threshold: Threshold for scene change detection (higher values mean fewer scenes)
        min_scene_duration: Minimum duration of a scene in seconds
        
    Returns:
        List of (start_time, end_time) tuples for detected scenes
    """
    # Open the video
    video = cv2.VideoCapture(str(video_path))
    
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Detecting scenes in video: {video_path} ({duration:.2f} seconds)")
    
    # Initialize variables
    prev_frame = None
    scenes = [(0.0, None)]  # List of (start_time, end_time) tuples
    frame_num = 0
    
    # Process frames
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compare with previous frame
        if prev_frame is not None:
            # Calculate mean absolute difference
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            
            # If the difference is above threshold, it's a new scene
            if mean_diff > threshold:
                # Get the time of this frame
                time = frame_num / fps
                
                # Check if the previous scene is long enough
                if time - scenes[-1][0] >= min_scene_duration:
                    # End the previous scene
                    scenes[-1] = (scenes[-1][0], time)
                    
                    # Start a new scene
                    scenes.append((time, None))
        
        # Update for next iteration
        prev_frame = gray
        frame_num += 1
    
    # End the last scene
    if scenes[-1][1] is None:
        scenes[-1] = (scenes[-1][0], duration)
    
    # Release the video
    video.release()
    
    logger.info(f"Detected {len(scenes)} scenes")
    return scenes

def track_faces(clip: VideoFileClip) -> List[Dict[str, Any]]:
    """
    Track faces in a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        
    Returns:
        List of face tracking data for each detected face
    """
    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        logger.warning("Failed to load face cascade classifier")
        return []
    
    # Initialize tracking data
    faces = []
    
    # Sample frames for tracking
    fps = clip.fps
    duration = clip.duration
    frame_count = int(duration * fps)
    
    # Sample every n frames
    sample_rate = max(1, int(fps / 4))  # 4 samples per second max
    
    for frame_idx in range(0, frame_count, sample_rate):
        time = frame_idx / fps
        
        if time >= duration:
            break
        
        # Get the frame at this time
        frame = clip.get_frame(time)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process detected faces
        for i, (x, y, w, h) in enumerate(detected_faces):
            # Calculate center position and size
            center_x = x + w / 2
            center_y = y + h / 2
            size = max(w, h)
            
            # Check if this is a new face or an existing one
            matched = False
            
            for face in faces:
                # Get the last position
                last_pos = face["positions"][-1] if face["positions"] else None
                
                if last_pos:
                    # Calculate distance to last position
                    dist = np.sqrt((center_x - last_pos["x"]) ** 2 + (center_y - last_pos["y"]) ** 2)
                    
                    # If the distance is small enough, this is the same face
                    if dist < size / 2:
                        # Add the new position
                        face["positions"].append({
                            "time": time,
                            "x": center_x,
                            "y": center_y,
                            "size": size
                        })
                        matched = True
                        break
            
            # If this is a new face, add it
            if not matched:
                faces.append({
                    "id": len(faces),
                    "positions": [{
                        "time": time,
                        "x": center_x,
                        "y": center_y,
                        "size": size
                    }]
                })
    
    # Filter out faces with too few positions
    faces = [face for face in faces if len(face["positions"]) >= 5]
    
    logger.info(f"Tracked {len(faces)} faces in video")
    return faces

def add_text_overlay(clip: VideoFileClip, 
                   text: str, 
                   position: Tuple[str, str] = ("center", "bottom"),
                   fontsize: int = 24,
                   color: str = "white",
                   bg_color: Optional[str] = "black",
                   bg_opacity: float = 0.5,
                   method: str = "caption") -> VideoFileClip:
    """
    Add text overlay to a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        text: Text to display
        position: (horizontal, vertical) position - options:
                  horizontal: "left", "center", "right"
                  vertical: "top", "center", "bottom"
        fontsize: Font size in points
        color: Text color
        bg_color: Background color (None for transparent)
        bg_opacity: Background opacity (0-1)
        method: Method to use - "caption" for simple text, "text" for more options
        
    Returns:
        Clip with text overlay
    """
    if method == "caption":
        result = clip.set_caption(
            text,
            position=position,
            fontsize=fontsize,
            color=color,
            bg_color=bg_color,
            background_opacity=bg_opacity
        )
    else:
        from moviepy.editor import TextClip
        
        # Create text clip
        txt_clip = TextClip(
            text,
            fontsize=fontsize,
            color=color,
            bg_color=bg_color,
            opacity=bg_opacity,
            method=method
        )
        
        # Position the text
        width, height = clip.size
        
        # Horizontal position
        if position[0] == "left":
            x_pos = 10
        elif position[0] == "center":
            x_pos = width // 2
            txt_clip = txt_clip.set_position(("center", position[1]))
        else:  # right
            x_pos = width - 10
            txt_clip = txt_clip.set_position(("right", position[1]))
        
        # Vertical position
        if position[1] == "top":
            y_pos = 10
        elif position[1] == "center":
            y_pos = height // 2
        else:  # bottom
            y_pos = height - 10 - txt_clip.h
        
        # Set position if not already set
        if position[0] != "center":
            txt_clip = txt_clip.set_position((x_pos, y_pos))
        
        # Compose with original clip
        result = CompositeVideoClip([clip, txt_clip])
    
    return result

def add_subtitles(clip: VideoFileClip, 
                transcription: Transcription,
                fontsize: int = 24,
                color: str = "white",
                bg_color: str = "black",
                bg_opacity: float = 0.5) -> VideoFileClip:
    """
    Add subtitles to a video clip based on transcription.
    
    Args:
        clip: MoviePy VideoFileClip object
        transcription: Transcription object
        fontsize: Font size in points
        color: Text color
        bg_color: Background color
        bg_opacity: Background opacity (0-1)
        
    Returns:
        Clip with subtitles
    """
    from moviepy.editor import TextClip, CompositeVideoClip
    
    # Create a list of subtitle clips
    subtitle_clips = []
    
    for segment in transcription.segments:
        # Create text clip for this subtitle
        txt_clip = TextClip(
            segment.text,
            fontsize=fontsize,
            color=color,
            bg_color=bg_color,
            method='caption',
            size=(clip.w * 0.9, None),
            align='center'
        )
        
        # Set position at bottom center
        txt_clip = txt_clip.set_position(('center', 'bottom'))
        
        # Set duration and start time
        txt_clip = txt_clip.set_start(segment.start).set_duration(segment.duration())
        
        # Set opacity for background
        if bg_opacity < 1.0:
            txt_clip = txt_clip.set_opacity(bg_opacity)
        
        # Add to list
        subtitle_clips.append(txt_clip)
    
    # Composite all clips
    result = CompositeVideoClip([clip] + subtitle_clips)
    return result

def adjust_speed(clip: VideoFileClip, 
               speed_factor: float = 1.0,
               preserve_pitch: bool = True) -> VideoFileClip:
    """
    Adjust the speed of a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        speed_factor: Speed multiplier (1.0 = normal speed, 2.0 = twice as fast)
        preserve_pitch: Whether to preserve audio pitch when changing speed
        
    Returns:
        Speed-adjusted clip
    """
    # Apply speed change to video
    adjusted_clip = clip.fx(vfx.speedx, speed_factor)
    
    # For audio pitch preservation (if requested and has audio)
    if preserve_pitch and clip.audio is not None and speed_factor != 1.0:
        # This is a simplified approach - for proper pitch preservation,
        # you'd need more sophisticated audio processing
        # We're just applying the speedx effect to audio
        adjusted_clip = adjusted_clip.set_audio(clip.audio.fx(vfx.speedx, speed_factor))
    
    return adjusted_clip

def add_watermark(clip: VideoFileClip, 
                watermark_image: Union[str, Path],
                position: Tuple[str, str] = ("right", "bottom"),
                opacity: float = 0.5,
                margin: int = 10) -> VideoFileClip:
    """
    Add a watermark to a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        watermark_image: Path to watermark image
        position: (horizontal, vertical) position - options:
                  horizontal: "left", "center", "right"
                  vertical: "top", "center", "bottom"
        opacity: Watermark opacity (0-1)
        margin: Margin from edge in pixels
        
    Returns:
        Clip with watermark
    """
    from moviepy.editor import ImageClip
    
    # Load the watermark image
    watermark = ImageClip(str(watermark_image))
    
    # Set opacity
    watermark = watermark.set_opacity(opacity)
    
    # Calculate position
    width, height = clip.size
    wm_width, wm_height = watermark.size
    
    # Horizontal position
    if position[0] == "left":
        x_pos = margin
    elif position[0] == "center":
        x_pos = (width - wm_width) // 2
    else:  # right
        x_pos = width - wm_width - margin
    
    # Vertical position
    if position[1] == "top":
        y_pos = margin
    elif position[1] == "center":
        y_pos = (height - wm_height) // 2
    else:  # bottom
        y_pos = height - wm_height - margin
    
    # Set position
    watermark = watermark.set_position((x_pos, y_pos))
    
    # Composite with original clip
    result = CompositeVideoClip([clip, watermark])
    return result

def trim_silence(clip: VideoFileClip, 
               min_silence_duration: float = 1.0,
               silence_threshold: float = 0.05) -> VideoFileClip:
    """
    Trim silent portions from a video clip.
    
    Args:
        clip: MoviePy VideoFileClip object
        min_silence_duration: Minimum duration of silence to trim (seconds)
        silence_threshold: Threshold for considering audio as silence (0-1)
        
    Returns:
        Clip with silent portions removed
    """
    # Extract audio to a temporary file
    from .transcription import find_silent_segments, extract_audio_from_video
    
    # Create a temporary file for the audio
    fd, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    try:
        # Write audio to the temporary file
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path, logger=None)
            
            # Find silent segments
            silent_segments = find_silent_segments(
                audio_path, 
                min_silence_duration, 
                silence_threshold
            )
            
            if not silent_segments:
                return clip
            
            # Create a list of segments to keep
            segments_to_keep = []
            current_pos = 0
            
            for silent_start, silent_end in silent_segments:
                # Add segment before silence
                if silent_start > current_pos:
                    segments_to_keep.append((current_pos, silent_start))
                
                # Update position to after silence
                current_pos = silent_end
            
            # Add final segment after last silence
            if current_pos < clip.duration:
                segments_to_keep.append((current_pos, clip.duration))
            
            # If no segments to keep, return original clip
            if not segments_to_keep:
                return clip
            
            # Create a new clip with concatenated segments
            from moviepy.editor import concatenate_videoclips
            
            subclips = [clip.subclip(start, end) for start, end in segments_to_keep]
            result = concatenate_videoclips(subclips)
            
            return result
        else:
            # No audio, return original clip
            return clip
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(audio_path)
        except (OSError, FileNotFoundError):
            pass 