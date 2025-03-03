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
import mediapipe as mp  # Add MediaPipe import
from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip, 
    ColorClip, ImageClip, concatenate_videoclips, VideoClip
)
from tqdm import tqdm
import zhconv  # For converting between Chinese variants
import requests
import openai
from PIL import Image, ImageDraw, ImageFont
import jieba  # Add this to the imports at the top

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
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.Client(api_key=OPENAI_API_KEY)

class TranscriptionSegment:
    """Represents a transcription segment with timing information and word-level timestamps"""
    def __init__(self, text: str, start: float, end: float, words: Optional[List[Dict[str, Any]]] = None):
        self.text = text
        self.start = start
        self.end = end
        self.duration = end - start
        self.words = words or []  # List of word timestamps {text, start, end}

    def __str__(self):
        return f"{self.start:.2f}s - {self.end:.2f}s: {self.text}"

class HighlightClip:
    """Represents a potential highlight clip with score and timing information"""
    def __init__(self, start: float, end: float, score: float, title: str):
        self.start = start
        self.end = end
        self.duration = end - start
        self.score = score
        self.title = title

    def __str__(self):
        return f"Highlight: {self.start:.2f}s - {self.end:.2f}s ({self.duration:.2f}s) | Score: {self.score} | Title: {self.title}"

class FontManager:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        self.font_path = os.path.join(self.font_dir, 'NotoSansTC-Regular.otf')
        self._ensure_font_exists()
    
    def _ensure_font_exists(self):
        """Ensure the Chinese font exists, download if not present."""
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
        
        if not os.path.exists(self.font_path):
            logger.info("Downloading Chinese font...")
            font_url = 'https://fonts.gstatic.com/s/notosanstc/v26/-nF7OG829Oofr2wohFbTp9iFOSsLA_ZJ1g.otf'
            try:
                response = requests.get(font_url)
                response.raise_for_status()
                with open(self.font_path, 'wb') as f:
                    f.write(response.content)
                # Verify font file
                try:
                    ImageFont.truetype(self.font_path, 16)
                    logger.info("Font downloaded and verified successfully!")
                except Exception as e:
                    logger.error(f"Downloaded font file is invalid: {e}")
                    if os.path.exists(self.font_path):
                        os.remove(self.font_path)
                    raise
            except Exception as e:
                logger.error(f"Failed to download font: {e}")
                raise
    
    def get_font_path(self):
        """Get the path to the Chinese font."""
        return self.font_path

def transcribe_video(video_path: str, language_code: str = "zh") -> List[TranscriptionSegment]:
    """
    Transcribe the video using AssemblyAI and return segments with timing information.
    First extracts audio from video, then transcribes the audio.
    Uses word-level timestamps for more precise timing information.
    If a transcript file already exists, load it instead of re-transcribing.
    
    Args:
        video_path: Path to the video file
        language_code: Language code (zh for Traditional Chinese)
    
    Returns:
        List of TranscriptionSegment objects
    """
    # Create transcript file path
    video_path_obj = Path(video_path)
    transcript_path = video_path_obj.with_suffix('.transcript.json')
    
    # Check if transcript file exists
    if transcript_path.exists():
        logger.info(f"Loading existing transcript from {transcript_path}")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Reconstruct segments from saved data
            segments = []
            for seg_data in transcript_data:
                segments.append(TranscriptionSegment(
                    text=seg_data['text'],
                    start=seg_data['start'],
                    end=seg_data['end'],
                    words=seg_data.get('words', [])  # Load word-level timestamps if available
                ))
            logger.info(f"Loaded {len(segments)} segments from existing transcript")
            return segments
        except Exception as e:
            logger.warning(f"Failed to load existing transcript: {e}. Will re-transcribe.")
    
    logger.info(f"Extracting audio from video: {video_path}")
    
    # Extract audio to temporary file
    temp_audio_path = video_path_obj.with_suffix('.temp.wav')
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        video.close()
        
        logger.info(f"Transcribing audio...")
        
        # Create the transcriber with configuration
        config = aai.TranscriptionConfig(
            language_code=language_code,
            speaker_labels=True,
            punctuate=True,
            format_text=True
        )
        transcriber = aai.Transcriber()
        
        # Start the transcription with audio file
        transcript = transcriber.transcribe(str(temp_audio_path), config=config)
        
        # Show progress bar while waiting for transcription
        with tqdm(total=100, desc="Transcribing") as pbar:
            last_progress = 0
            while True:
                status = transcript.status
                
                # Calculate progress percentage based on status
                if status == "queued":
                    progress = 5
                elif status == "processing":
                    progress = 50
                elif status == "completed":
                    progress = 100
                    break
                elif status == "error":
                    raise Exception(f"Transcription failed with status: {status}")
                
                # Update progress bar
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
                
                time.sleep(3)  # Wait 3 seconds before checking again
        
        # Process word-level timestamps and group into segments
        segments = []
        current_segment = []
        current_words = []
        current_start = None
        
        # Process each word with its timestamp
        for word in transcript.words:
            if not current_start:
                current_start = word.start / 1000  # Convert ms to seconds
            
            word_info = {
                'text': word.text,
                'start': word.start / 1000,  # Convert ms to seconds
                'end': word.end / 1000       # Convert ms to seconds
            }
            
            current_segment.append(word.text)
            current_words.append(word_info)
            
            # Get the current text accumulated so far
            current_text = ' '.join(current_segment)
            
            # Create a new segment when we hit any of these conditions:
            # 1. Natural sentence ending punctuation
            # 2. Long pause (> 1.5 seconds)
            # 3. Sentence is getting too long (> 50 characters)
            # 4. Natural break point detected by jieba
            should_break = False
            
            # Check for sentence ending punctuation
            if any(p in word.text for p in '.!?。！？'):
                should_break = True
            
            # Check for long pause (if not the first word)
            elif len(current_words) > 1:
                pause_duration = word_info['start'] - current_words[-2]['end']
                if pause_duration > 1.5:  # 1.5 second pause threshold
                    should_break = True
            
            # Check sentence length
            elif len(current_text) > 50:
                # Use jieba to find a good break point
                words_list = list(jieba.cut(current_text))
                if len(words_list) > 1:  # If we can actually split it
                    should_break = True
            
            # Check for natural break using jieba
            elif len(current_text) > 15:  # Only check if we have enough text
                words_list = list(jieba.cut(current_text))
                # Break if we detect a complete phrase or clause
                if any(w in ['的', '了', '和', '與', '但是', '所以', '因為', '如果'] for w in words_list[-2:]):
                    should_break = True
            
            if should_break:
                text = ' '.join(current_segment)
                text = zhconv.convert(text, 'zh-hant')  # Convert to traditional Chinese
                segments.append(TranscriptionSegment(
                    text=text,
                    start=current_start,
                    end=word.end / 1000,  # Convert ms to seconds
                    words=current_words
                ))
                current_segment = []
                current_words = []
                current_start = None
        
        # Add any remaining words as a segment
        if current_segment:
            text = ' '.join(current_segment)
            text = zhconv.convert(text, 'zh-hant')  # Convert to traditional Chinese
            segments.append(TranscriptionSegment(
                text=text,
                start=current_start,
                end=transcript.words[-1].end / 1000,  # Convert ms to seconds
                words=current_words
            ))
        
        # Save transcript to file with word-level timestamps
        logger.info(f"Saving transcript to {transcript_path}")
        transcript_data = [
            {
                'text': seg.text,
                'start': seg.start,
                'end': seg.end,
                'words': seg.words  # Include word-level timestamps
            }
            for seg in segments
        ]
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcription complete: {len(segments)} segments")
        return segments
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.info("Cleaned up temporary audio file")

def identify_highlights(segments: List[TranscriptionSegment], max_duration: float = 120.0) -> List[Dict[str, Any]]:
    """
    Two-step approach:
    1. Use OpenAI o3-mini to identify one continuous engaging segment of ~120 seconds
    2. Use Claude to format the response into proper JSON
    
    Args:
        segments: List of transcription segments
        max_duration: Target duration for the highlight clip in seconds (default: 120s)
    
    Returns:
        List of dictionaries containing segment information
    """
    logger.info("Step 1: Identifying one continuous engaging segment using OpenAI o3-mini")
    
    # Format the transcription for OpenAI
    transcript_text = "\n".join([f"{i+1}. {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}" for i, seg in enumerate(segments)])
    
    # Save transcript text for debugging
    debug_file = Path("openai_debug.log")
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write("=== Transcript Text ===\n")
        f.write(transcript_text)
        f.write("\n\n")
    
    try:
        # Log OpenAI request
        logger.debug("Sending request to OpenAI API")
        logger.debug("Request payload:")
        request_payload = {
            "model": "o3-mini",
            "messages": [
                {
                    "role": "developer",
                    "content": "You are a video editing expert. Your task is to analyze video transcripts and identify one continuous engaging segment that is approximately 120 seconds long. Focus on finding a segment that is self-contained, has natural start/end points, and maintains narrative coherence. Do not include double quotes in the response."
                },
                {
                    "role": "system",
                    "content": "Analyze the transcript and identify one continuous engaging segment that is approximately 120 seconds long. Ensure the segment has clear start/end times and maintains context."
                },
                {
                    "role": "user",
                    "content": f"""以下是視頻的完整逐字稿（包含時間戳）：

{transcript_text}

請仔細分析這些內容，然後：
1. 先看整個視頻，了解整個視頻的內容和主題
2. 找出一個大約120秒長的連續精彩片段
3. 片段的開始和結束要在自然的開始和結束，不要牽強或者突然地開始和結束

請列出：
1. 整個視頻的主題
2. 精彩片段的：
   - 開始時間（秒）
   - 結束時間（秒）
   - 內容摘要
   - 選擇原因"""
                }
            ],
        }
        logger.debug(json.dumps(request_payload, ensure_ascii=False, indent=2))
        
        # Append request to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Request ===\n")
            f.write(json.dumps(request_payload, ensure_ascii=False, indent=2))
            f.write("\n\n")
        
        # Call OpenAI API
        openai_response = openai_client.chat.completions.create(**request_payload)
        
        # Log OpenAI response
        logger.debug("OpenAI API Response:")
        logger.debug(str(openai_response))
        
        # Append response to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Response ===\n")
            f.write(str(openai_response))
            f.write("\n\n")
        
        # Get OpenAI's analysis
        openai_analysis = openai_response.choices[0].message.content
        logger.debug("OpenAI Analysis:")
        logger.debug(openai_analysis)
        
        # Append analysis to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Analysis ===\n")
            f.write(openai_analysis)
            f.write("\n\n")
        
        logger.info("Step 2: Formatting response using Claude")
        
        # Prepare prompt for Claude to format the response
        claude_prompt = f"""Please convert the following video analysis into a properly formatted JSON structure. Escape all double quotes in the JSON.

Analysis from OpenAI:
{openai_analysis}

Please format it as:
```json
{{
  "segments": [
    {{
      "start_time": start_time_in_seconds,
      "end_time": end_time_in_seconds,
      "content": "content_summary",
      "reason": "selection_reason"
    }}
  ],
  "theme": "video_theme"
}}
```

Ensure all times are in seconds (floating point numbers) and the content and reasons are in Traditional Chinese."""

        # Log Claude prompt
        logger.debug("Claude Prompt:")
        logger.debug(claude_prompt)
        
        # Append Claude prompt to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Claude Prompt ===\n")
            f.write(claude_prompt)
            f.write("\n\n")
        
        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            temperature=0.2,
            max_tokens=1000,
            system="You are a helpful assistant that converts video analysis into properly formatted JSON. Always maintain Traditional Chinese text in the output.",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        
        # Log Claude response
        response_text = response.content[0].text
        logger.debug("Claude Response:")
        logger.debug(response_text)
        
        # Append Claude response to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Claude Response ===\n")
            f.write(response_text)
            f.write("\n\n")
        
        # Extract JSON from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Log final formatted data
            logger.debug("Final Formatted Data:")
            logger.debug(json.dumps(data, ensure_ascii=False, indent=2))
            
            # Append final data to debug file
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== Final Formatted Data ===\n")
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
                f.write("\n\n")
            
            return data
        else:
            error_msg = "Failed to extract JSON from Claude response"
            logger.error(error_msg)
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== Error ===\n")
                f.write(error_msg)
                f.write("\n\n")
            return None
    
    except Exception as e:
        error_msg = f"Error identifying segments: {str(e)}"
        logger.error(error_msg)
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Error ===\n")
            f.write(error_msg)
            f.write("\n\n")
        return None

def generate_titles(segments_data: Dict[str, Any]) -> Tuple[str, List[HighlightClip]]:
    """
    Use Claude 3.5 to generate engaging titles for the segments
    
    Args:
        segments_data: Dictionary containing segment information and theme
    
    Returns:
        Tuple of (main_title, List[HighlightClip])
    """
    logger.info("Generating segment title using Claude")
    
    # Prepare prompt for Claude
    segments_text = "\n".join([
        f"{i+1}. {seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n內容：{seg['content']}\n原因：{seg['reason']}"
        for i, seg in enumerate(segments_data['segments'])
    ])
    
    prompt = f"""你是影片標題專家，需要為一個視頻片段創作吸引人的標題。要直接、簡潔、有力，不要使用過多的修飾詞。要严格符合圣经教义.

視頻主題：{segments_data['theme']}

精彩片段內容：
{segments_text}

請為片段創作標題，格式如下：
```json
{{
  "segments": [
    {{
      "title": "片段標題（15字以內，繁體中文）",
      "score": 吸引力評分（0-100）
    }}
  ]
}}
```

要求：
1. 片段標題要吸引人，突出該片段最精彩的部分
2. 所有標題使用繁體中文
3. 標題要簡潔有力，容易理解"""

    try:
        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system="You are a creative title generator specializing in engaging video titles. Always respond in Traditional Chinese.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from the response
        response_text = response.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Create HighlightClip objects
            highlights = []
            for seg_data, title_data in zip(segments_data['segments'], data['segments']):
                highlight = HighlightClip(
                    start=float(seg_data['start_time']),
                    end=float(seg_data['end_time']),
                    score=float(title_data['score']),
                    title=title_data['title']
                )
                highlights.append(highlight)
            
            # Return empty string for main_title since we're not generating it
            return "", highlights
        else:
            logger.error("Failed to extract JSON from Claude response")
            return "", []
    
    except Exception as e:
        logger.error(f"Error generating titles: {e}")
        return "", []

def create_layout_preview(
    input_path: str,
    timestamp: float,
    main_title: str,
    output_preview_path: str,
    logo_path: Optional[str] = None,
    segments: Optional[List[TranscriptionSegment]] = None,  # Add segments parameter
    highlight: Optional[HighlightClip] = None  # Add highlight parameter
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
    
    # Create PIL Image for preview
    frame_image = Image.fromarray(frame)
    
    # Calculate dimensions for 9:16 format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Create black background
    preview = Image.new('RGB', (output_width, output_height), 'black')
    
    # Resize frame while maintaining aspect ratio
    frame_width = output_width
    frame_height = int(frame_image.height * (output_width / frame_image.width))
    frame_resized = frame_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    logger.info(f"Resized video frame dimensions: {frame_width}x{frame_height} (width x height)")
    
    # Calculate position to center the frame
    frame_y = (output_height - frame_height) // 2
    logger.info(f"Video frame Y position: {frame_y} (distance from top)")
    preview.paste(frame_resized, (0, frame_y))
    
    # Create draw object
    draw = ImageDraw.Draw(preview)
    
    # Load fonts with proper Chinese support
    title_font = ImageFont.truetype(font_path, 80)
    church_font = ImageFont.truetype(font_path, 50)
    service_font = ImageFont.truetype(font_path, 40)
    logger.info("Using Noto Sans TC font with sizes: title=80px, church=50px, service=40px")
    
    # Define text areas with increased space for logo
    title_margin = 40
    church_name_height = 130  # Increased from 100 to 200 for larger logo
    main_title_height = 150
    service_info_height = 80
    
    # Debug text area dimensions
    logger.info("Text area dimensions:")
    logger.info(f"- Title margin: {title_margin}px")
    logger.info(f"- Church name height: {church_name_height}px")
    logger.info(f"- Main title height: {main_title_height}px")
    logger.info(f"- Service info height: {service_info_height}px")
    
    # Calculate and debug text positions
    church_name_y = title_margin
    main_title_y = title_margin + church_name_height + main_title_height//2
    service_info_y = title_margin + church_name_height + main_title_height + service_info_height//2
    
    logger.info("Text Y positions from top:")
    logger.info(f"- Church name area: {church_name_y}px")
    logger.info(f"- Main title: {main_title_y}px")
    logger.info(f"- Service info: {service_info_y}px")
    
    # Add logo for church name with increased size
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            # Calculate logo size to fit church name height while maintaining aspect ratio
            logo_height = church_name_height
            logo_width = int(logo.width * (logo_height / logo.height))
            
            # Allow logo to be wider, up to 90% of output width
            max_logo_width = int(output_width * 0.9)  # Increased from fixed margin to 90% of width
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
            logger.info(f"Falling back to text: '{church_name}' at y={church_name_y + church_name_height//2}")
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
        logger.info(f"No logo provided, using text: '{church_name}' at y={church_name_y + church_name_height//2}")
        draw.text(
            (output_width//2, church_name_y + church_name_height//2),
            church_name,
            font=church_font,
            fill='white',
            anchor='mm'
        )
    
    # Draw main title
    logger.info(f"Drawing main title: '{main_title}' at y={main_title_y}")
    draw.text(
        (output_width//2, main_title_y),
        main_title,
        font=title_font,
        fill='#00FF00',
        anchor='mm'
    )
    
    # Draw service info
    service_info = "主日崇拜: 每週日下午2点"
    logger.info(f"Drawing service info: '{service_info}' at y={service_info_y}")
    draw.text(
        (output_width//2, service_info_y),
        service_info,
        font=service_font,
        fill='white',
        anchor='mm'
    )
    
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
            subtitle_height = 120  # Match the height used in create_highlight_video
            subtitle_fontsize = 65  # Match the fontsize used in create_highlight_video
            subtitle_margin = 40
            
            # Position subtitle below video frame
            subtitle_y = frame_y + frame_height + subtitle_margin
            
            # Create subtitle font
            subtitle_font = ImageFont.truetype(font_path, subtitle_fontsize)
            
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
    print(f"- Video frame: {frame_width}x{frame_height} at y={frame_y}")
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

def create_highlight_video(
    input_path: str, 
    highlight: HighlightClip, 
    segments: List[TranscriptionSegment],
    output_path: str,
    main_title: str,  # This will be ignored as we'll use highlight.title
    logo_path: Optional[str] = None,
    add_subtitles: bool = False
):
    """
    Create highlight video in 9:16 format with titles and subtitles, styled like a church service video
    First creates a preview for confirmation, then proceeds with video creation if approved.
    Uses the segment title as the main title text.
    The main video section will be in 1:1 ratio.
    
    Args:
        input_path: Path to input video
        highlight: HighlightClip object with timing information
        segments: List of transcription segments
        output_path: Path to output video
        main_title: Ignored as we use highlight.title instead
        logo_path: Path to logo image file (optional)
        add_subtitles: Whether to add subtitles to the video (default: False)
    """
    # Create preview first
    preview_path = str(Path(output_path).with_suffix('.preview.jpg'))
    preview_timestamp = highlight.start + (highlight.end - highlight.start) / 2  # Use middle frame for preview
    
    if not create_layout_preview(
        input_path=input_path,
        timestamp=preview_timestamp,
        main_title=highlight.title,
        output_preview_path=preview_path,
        logo_path=logo_path,
        segments=segments,
        highlight=highlight
    ):
        logger.info("Video creation cancelled by user")
        return

    logger.info(f"Creating highlight video from {highlight.start:.2f}s to {highlight.end:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    
    # Load video clip and apply zoom and tracking
    video = VideoFileClip(input_path).subclip(highlight.start, highlight.end)
    logger.info(f"Original video dimensions: {video.size[0]}x{video.size[1]} (width x height)")
    
    # Apply zoom and tracking
    logger.info("Applying 200% zoom and horizontal person tracking...")
    processed_clip = process_video_with_zoom_and_tracking(video, zoom_factor=2.0)
    
    # Calculate dimensions for 9:16 portrait format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Calculate dimensions for 1:1 video section
    video_section_size = output_width  # Square video section
    logger.info(f"Video section dimensions (1:1): {video_section_size}x{video_section_size} (width x height)")
    
    # Resize video while maintaining aspect ratio to fit the square
    orig_height, orig_width = processed_clip.size[1], processed_clip.size[0]
    if orig_width > orig_height:
        # For landscape video, fit to height
        scale_factor = video_section_size / orig_height
        scaled_width = int(orig_width * scale_factor)
        scaled_height = video_section_size
        # Crop excess width
        x_offset = (scaled_width - video_section_size) // 2
        processed_clip = processed_clip.resize(width=scaled_width, height=scaled_height)
        processed_clip = processed_clip.crop(x1=x_offset, y1=0, x2=x_offset + video_section_size, y2=video_section_size)
    else:
        # For portrait video, fit to width
        scale_factor = video_section_size / orig_width
        scaled_width = video_section_size
        scaled_height = int(orig_height * scale_factor)
        # Crop excess height
        y_offset = (scaled_height - video_section_size) // 2
        processed_clip = processed_clip.resize(width=scaled_width, height=scaled_height)
        processed_clip = processed_clip.crop(x1=0, y1=y_offset, x2=video_section_size, y2=y_offset + video_section_size)
    
    logger.info(f"Final video dimensions: {video_section_size}x{video_section_size} (width x height)")
    
    # Create a black background for the 9:16 format
    black_bg = ColorClip(size=(output_width, output_height), color=(0, 0, 0))
    black_bg = black_bg.set_duration(video.duration)
    
    # Position the video in the center vertically with more space for the square
    # Calculate positions to center the 1:1 video with proper margins
    title_margin = 40
    church_name_height = 200  # Increased from 100 to 200 for larger logo
    main_title_height = 150
    service_info_height = 80
    
    # Calculate total height of text elements
    total_text_height = (title_margin + church_name_height + main_title_height + 
                        service_info_height + title_margin)
    
    # Position video after the top text elements with some margin
    video_y = total_text_height + title_margin
    video_pos = ('center', video_y)
    logger.info(f"Video position: center, y={video_y} (distance from top)")
    
    # Initialize clips list
    clips_to_combine = [black_bg, processed_clip.set_position(video_pos)]
    
    # Calculate text positions relative to the new video position
    church_name_y = title_margin
    main_title_y = title_margin + church_name_height + main_title_height//2
    service_info_y = title_margin + church_name_height + main_title_height + service_info_height//2
    
    # Add logo or church name with increased size
    if logo_path and os.path.exists(logo_path):
        try:
            # Create logo clip
            logo_clip = ImageClip(logo_path)
            
            # Calculate logo size with increased height
            logo_height = church_name_height
            logo_width = int(logo_clip.size[0] * (logo_height / logo_clip.size[1]))
            
            # Allow logo to be wider, up to 90% of output width
            max_logo_width = int(output_width * 0.9)  # Increased from fixed margin to 90% of width
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
    
    # Add main title
    main_title_clip = TextClip(
        highlight.title,
        fontsize=80,
        color='#00FF00',
        font=font_path,
        size=(output_width - 2*title_margin, main_title_height)
    ).set_position(('center', main_title_y))
    main_title_clip = main_title_clip.set_duration(video.duration)
    clips_to_combine.append(main_title_clip)
    logger.info(f"Added main title clip: '{highlight.title}'")
    
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
    
    # Add subtitles only if requested
    if add_subtitles:
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

def process_video_with_zoom_and_tracking(clip: VideoFileClip, zoom_factor: float = 2.0) -> VideoFileClip:
    """
    Process video with static zoom and horizontal person tracking.
    Centers the person in frame while respecting video boundaries.
    Uses responsive tracking without speed limits for immediate following.
    
    Args:
        clip: Input VideoFileClip
        zoom_factor: Static zoom factor (default: 2.0 for 200% zoom)
    
    Returns:
        Processed VideoFileClip with zoom and tracking
    """
    # Initialize MediaPipe Pose with global variable to prevent recreation
    global mp_pose_instance
    if 'mp_pose_instance' not in globals():
        mp_pose = mp.solutions.pose
        mp_pose_instance = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    # Store previous positions for smooth tracking
    prev_center_x = None
    prev_offset_x = None
    smoothing_factor = 0.3  # Increased for more immediate response
    
    def get_person_center(landmarks, frame_width, frame_height):
        """Calculate the center point of the person using multiple body landmarks"""
        # Key points for center calculation
        key_points = [
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.NOSE
        ]
        
        points = []
        for point in key_points:
            landmark = landmarks.landmark[point]
            if landmark.visibility > 0.5:  # Only use visible landmarks
                points.append(landmark.x)
        
        if points:
            # Use average of visible points
            return sum(points) / len(points) * frame_width
        else:
            # Fallback to frame center if no points are visible
            return frame_width / 2

    def process_frame(frame):
        nonlocal prev_center_x, prev_offset_x
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = mp_pose_instance.process(frame_rgb)
            
            # Calculate zoomed dimensions
            zoomed_width = int(frame_width * zoom_factor)
            zoomed_height = int(frame_height * zoom_factor)
            
            # Create zoomed frame
            zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
            
            # Maximum possible offset (half of the extra width from zooming)
            max_offset_x = (zoomed_width - frame_width) // 2
            
            if results and results.pose_landmarks:
                # Calculate person center using multiple body points
                current_center_x = get_person_center(results.pose_landmarks, frame_width, frame_height)
                
                # Apply light smoothing to center position
                if prev_center_x is not None:
                    person_center_x = prev_center_x + smoothing_factor * (current_center_x - prev_center_x)
                else:
                    person_center_x = current_center_x
                
                # Update previous center
                prev_center_x = person_center_x
                
                # Calculate target offset to center the person
                # Positive offset moves frame left, negative moves right
                target_offset_x = (frame_width/2 - person_center_x) * 1.2  # 1.2 factor for more aggressive centering
                
                # Limit target offset to valid range
                target_offset_x = np.clip(target_offset_x, -max_offset_x, max_offset_x)
                
                # Apply smoothing to camera movement
                if prev_offset_x is not None:
                    offset_x = prev_offset_x + smoothing_factor * (target_offset_x - prev_offset_x)
                else:
                    offset_x = target_offset_x
                
                # Ensure offset stays within bounds
                offset_x = np.clip(offset_x, -max_offset_x, max_offset_x)
                
                # Update previous offset
                prev_offset_x = offset_x
                
            else:
                # If no person detected, gradually return to center
                if prev_offset_x is not None:
                    # Quick return to center when no person detected
                    target_offset_x = 0
                    offset_x = prev_offset_x + smoothing_factor * 2 * (target_offset_x - prev_offset_x)  # Faster return
                    prev_offset_x = offset_x
                else:
                    offset_x = 0
            
            # Calculate crop coordinates
            # Vertical center crop (static)
            start_y = (zoomed_height - frame_height) // 2
            end_y = start_y + frame_height
            
            # Horizontal crop based on offset
            start_x = int((zoomed_width - frame_width) // 2 - offset_x)
            end_x = start_x + frame_width
            
            # Final boundary check
            if start_x < 0:
                start_x = 0
                end_x = frame_width
            elif end_x > zoomed_width:
                end_x = zoomed_width
                start_x = zoomed_width - frame_width
            
            # Crop and return the frame
            return zoomed_frame[start_y:end_y, start_x:end_x]
            
        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            # Return centered crop of zoomed frame as fallback
            start_y = (zoomed_height - frame_height) // 2
            start_x = (zoomed_width - frame_width) // 2
            return zoomed_frame[start_y:start_y + frame_height, start_x:start_x + frame_width]
    
    # Process the clip
    processed_clip = clip.fl_image(process_frame)
    return processed_clip

def main():
    """Main function to process video and create highlight clips"""
    parser = argparse.ArgumentParser(description='Generate highlight clips from long videos')
    parser.add_argument('input_video', help='Path to the input video file')
    parser.add_argument('--output', '-o', help='Path to the output video file')
    parser.add_argument('--max-duration', '-d', type=float, default=120.0, 
                      help='Target duration for the highlight clip in seconds (default: 120)')
    parser.add_argument('--logo', '-l', 
                      default=os.path.join(os.path.dirname(__file__), 'default_assets', 'Waymaker_white_logo_transparent_background.png'),
                      help='Path to logo image file (default: default_assets/Waymaker_white_logo_transparent_background.png)')
    parser.add_argument('--resume-from', '-r', choices=['transcribe', 'segments', 'titles'],
                      help='Resume from a specific stage using saved state')
    parser.add_argument('--add-subtitles', action='store_true',
                      help='Add subtitles to the video (default: False)')
    
    args = parser.parse_args()
    
    # Default output path if not specified
    if not args.output:
        input_path = Path(args.input_video)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = str(input_path.with_name(f"{input_path.stem}_highlight_{timestamp}{input_path.suffix}"))
        logger.info(f"Setting default output path to: {args.output}")
    
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
        # Initialize state variables
        segments = None
        segments_data = None
        highlights = []
        highlight = None
        in_title_generation = False

        # Step 1: Load or generate transcription
        if args.resume_from in ['transcribe', 'segments', 'titles']:
            # Load transcription state
            transcript_data = load_state(args.input_video, 'transcript')
            if transcript_data:
                segments = [
                    TranscriptionSegment(
                        text=seg['text'],
                        start=seg['start'],
                        end=seg['end'],
                        words=seg.get('words', [])
                    )
                    for seg in transcript_data
                ]
                logger.info(f"Loaded transcription with {len(segments)} segments")
            else:
                logger.error("No transcription state found to resume from")
                return
        else:
            segments = transcribe_video(args.input_video)
            # Save transcription state
            save_state(args.input_video, 'transcript', [
                {
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'words': seg.words
                }
                for seg in segments
            ])

        # Step 2: Load or identify segments
        if args.resume_from in ['segments', 'titles']:
            # Load segments state
            segments_data = load_state(args.input_video, 'segments')
            if not segments_data:
                logger.error("No segments state found to resume from")
                return
            logger.info("Loaded segments data from state")

        # Step 3: Load titles if resuming from titles
        if args.resume_from == 'titles':
            # Load titles state
            titles_data = load_state(args.input_video, 'titles')
            if titles_data and titles_data.get('segments'):
                title_data = titles_data['segments'][0]  # We now only have one highlight
                # Create highlight object from the saved state
                seg_data = segments_data['segments'][0]
                highlight = HighlightClip(
                    start=float(seg_data['start_time']),
                    end=float(seg_data['end_time']),
                    score=float(title_data['score']),
                    title=title_data['title']
                )
                highlights = [highlight]
                in_title_generation = True
                logger.info("Loaded titles data from state")
            else:
                logger.error("No titles state found to resume from")
                return

        while True:
            # Main workflow loop
            if not in_title_generation:
                # Show segment details and ask for confirmation
                if not segments_data:
                    segments_data = identify_highlights(segments, args.max_duration)
                    if not segments_data:
                        logger.error("No segment identified")
                        return
                    # Save segments state
                    save_state(args.input_video, 'segments', segments_data)
                
                print("\nSelected Segment:")
                print("-" * 50)
                
                seg = segments_data['segments'][0]  # We now only have one segment
                start_time = float(seg['start_time'])
                end_time = float(seg['end_time'])
                duration = end_time - start_time
                
                print(f"Time Range: {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)")
                print(f"Content Summary: {seg['content']}")
                print(f"Selection Reason: {seg['reason']}")
                
                # Show original transcripts for this time range
                print("\nOriginal Transcripts:")
                print("-" * 30)
                for transcript_seg in segments:
                    if (transcript_seg.start <= end_time and transcript_seg.end >= start_time):
                        print(f"{transcript_seg.start:.2f}s - {transcript_seg.end:.2f}s: {transcript_seg.text}")
                
                print("\nVideo Theme:")
                print("-" * 50)
                print(segments_data['theme'])
                
                while True:
                    choice = input("\nDo you want to:\n1. Continue with this segment\n2. Generate new segment\n3. Exit\nEnter your choice (1/2/3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Please enter 1, 2, or 3.")
                
                if choice == '1':
                    # Move to title generation
                    in_title_generation = True
                    # Generate initial titles
                    _, highlights = generate_titles(segments_data)
                    if highlights:
                        highlight = highlights[0]  # We now only have one highlight
                        # Save titles state
                        save_state(args.input_video, 'titles', {
                            'segments': [
                                {
                                    'title': highlight.title,
                                    'score': highlight.score
                                }
                            ]
                        })
                    continue
                elif choice == '2':
                    # Generate new segment
                    logger.info("\nGenerating new segment...")
                    segments_data = identify_highlights(segments, args.max_duration)
                    if segments_data:
                        # Save segments state
                        save_state(args.input_video, 'segments', segments_data)
                        # Reset title generation variables
                        highlights = []
                        highlight = None
                    continue
                else:  # choice == '3'
                    print("Exiting program.")
                    return
            
            else:  # in title generation step
                if not highlight:
                    # Generate titles if we don't have them
                    _, highlights = generate_titles(segments_data)
                    if highlights:
                        highlight = highlights[0]  # We now only have one highlight
                        # Save titles state
                        save_state(args.input_video, 'titles', {
                            'segments': [
                                {
                                    'title': highlight.title,
                                    'score': highlight.score
                                }
                            ]
                        })
                    else:
                        logger.error("Failed to generate titles")
                        return
                
                print("\nGenerated Titles:")
                print("-" * 50)
                print(f"Segment Title: {highlight.title}")
                print(f"Engagement Score: {highlight.score}/100")
                
                while True:
                    title_choice = input("\nDo you want to:\n1. Continue with these titles\n2. Generate new titles\n3. Exit\nEnter your choice (1/2/3): ").strip()
                    if title_choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Please enter 1, 2, or 3.")
                
                if title_choice == '1':
                    # Create highlight video
                    create_highlight_video(
                        args.input_video,
                        highlight,
                        segments,
                        args.output,
                        highlight.title,  # Use highlight.title as the main title
                        args.logo,
                        args.add_subtitles  # Pass the subtitle flag
                    )
                    
                    logger.info(f"Successfully created highlight video: {args.output}")
                    return
                elif title_choice == '2':
                    # Regenerate titles using the same segments data
                    logger.info("Regenerating titles with existing segment data...")
                    _, highlights = generate_titles(segments_data)
                    if highlights:
                        highlight = highlights[0]  # We now only have one highlight
                        # Save titles state
                        save_state(args.input_video, 'titles', {
                            'segments': [
                                {
                                    'title': highlight.title,
                                    'score': highlight.score
                                }
                            ]
                        })
                    else:
                        logger.error("Failed to generate new titles")
                        return
                    continue
                else:  # title_choice == '3'
                    print("Exiting program.")
                    return
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise

if __name__ == "__main__":
    main() 