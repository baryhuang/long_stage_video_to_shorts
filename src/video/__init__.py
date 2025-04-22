#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video processing module for manipulating video frames and applying effects.
"""

from src.video.processor import track_and_zoom_video
from src.video.preview import create_zoom_preview
from src.video.video_processing import (
    apply_zoom_effect, 
    apply_pan_effect, 
    apply_fade_in_out,
    load_video,
    save_video,
    extract_clip
)

__all__ = [
    'track_and_zoom_video',
    'create_zoom_preview',
    'apply_zoom_effect',
    'apply_pan_effect',
    'apply_fade_in_out',
    'load_video',
    'save_video',
    'extract_clip'
] 