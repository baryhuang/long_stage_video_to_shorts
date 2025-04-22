#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcription module for generating and processing transcripts from videos/audio.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import assemblyai as aai
import zhconv
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

from src.models.transcription import TranscriptionSegment

logger = logging.getLogger(__name__)

def transcribe_video(video_path: str, language_code: str = "zh") -> List[TranscriptionSegment]:
    """
    Transcribe the video or audio file using AssemblyAI and return segments with timing information.
    First extracts audio from video if needed, then transcribes the audio.
    Uses word-level timestamps for more precise timing information.
    If a transcript file already exists, load it instead of re-transcribing.
    
    Args:
        video_path: Path to the video or audio file
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
                    words=seg_data.get('words', []),  # Include word-level timestamps
                    speaker=seg_data.get('speaker')  # Load speaker information
                ))
            logger.info(f"Loaded {len(segments)} segments from existing transcript")
            return segments
        except Exception as e:
            logger.warning(f"Failed to load existing transcript: {e}. Will re-transcribe.")
    
    logger.info(f"Processing file: {video_path}")
    
    # Extract audio to temporary file
    temp_audio_path = video_path_obj.with_suffix('.temp.wav')
    try:
        # Check if input is video or audio
        is_video = video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        
        if is_video:
            logger.info("Input is video, extracting audio...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
            video.close()
        else:
            logger.info("Input is audio, converting to WAV format...")
            audio = AudioFileClip(video_path)
            audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
            audio.close()
        
        logger.info(f"Transcribing audio with speaker diarization...")
        
        # Create the transcriber with configuration
        config = aai.TranscriptionConfig(
            language_code=language_code,
            speaker_labels=True,  # Enable speaker diarization
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
        
        # Process utterances with speaker information
        segments = []
        
        # First, process utterances to get speaker information
        logger.info("Processing speaker diarization results...")
        for utterance in transcript.utterances:
            # Convert text to traditional Chinese
            text = zhconv.convert(utterance.text, 'zh-hant')
            
            # Create segment with speaker information
            segment = TranscriptionSegment(
                text=text,
                start=utterance.start / 1000,  # Convert ms to seconds
                end=utterance.end / 1000,      # Convert ms to seconds
                speaker=utterance.speaker,
                words=[{
                'text': word.text,
                    'start': word.start / 1000,
                    'end': word.end / 1000
                } for word in utterance.words] if hasattr(utterance, 'words') else []
            )
            segments.append(segment)
        
        logger.info(f"Found {len(segments)} segments with {len(set(seg.speaker for seg in segments))} unique speakers")
        
        # Save transcript to file with speaker information
        logger.info(f"Saving transcript to {transcript_path}")
        transcript_data = [
            {
                'text': seg.text,
                'start': seg.start,
                'end': seg.end,
                'speaker': seg.speaker,
                'words': seg.words
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

def create_text_transcript(video_path: str, segments: List[TranscriptionSegment]) -> str:
    """
    Create a plain text transcript file from segments.
    
    Args:
        video_path: Path to the original video file
        segments: List of transcription segments
    
    Returns:
        Path to the created transcript file
    """
    video_path_obj = Path(video_path)
    transcript_path = video_path_obj.with_suffix('.transcript.txt')
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.start)
    
    # Write transcript to file
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for: {video_path_obj.name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Track current speaker for formatting
        current_speaker = None
        
        for i, seg in enumerate(sorted_segments, 1):
            # Format timestamp
            timestamp = f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}.{int((seg.start%1)*10):01d} - {int(seg.end//60):02d}:{int(seg.end%60):02d}.{int((seg.end%1)*10):01d}]"
            
            # Handle speaker changes
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                # Add a blank line before new speaker (except for first speaker)
                if i > 1:
                    f.write("\n")
                # Write speaker header
                if current_speaker:
                    f.write(f"\nSpeaker {current_speaker}:\n{'-' * 20}\n")
            
            # Write the segment with timestamp and text
            f.write(f"{timestamp} {seg.text}\n")
            
            # Add blank line every 5 segments within same speaker for readability
            if i % 5 == 0 and i < len(sorted_segments) and sorted_segments[i].speaker == current_speaker:
                f.write("\n")
    
    logger.info(f"Created plain text transcript at: {transcript_path}")
    return str(transcript_path) 