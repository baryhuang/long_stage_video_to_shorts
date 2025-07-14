# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application that automatically extracts engaging highlight clips from long videos and converts them to 9:16 portrait format suitable for social media. The system uses AI-powered transcription (AssemblyAI) and content analysis (OpenAI/Claude) to identify the most valuable segments, then applies smart video processing with face tracking and zooming.

## Key Commands

### Setup and Environment
```bash
# Initial setup (creates venv, installs dependencies, creates .env)
chmod +x setup.sh && ./setup.sh

# Activate virtual environment
source venv/bin/activate

# Install dependencies manually
pip install -r requirements.txt

# Test setup and verify API keys
python test_setup.py
```

### Main Usage
```bash
# Basic usage - process single video
python highlight_generator.py --input-file video.mp4

# Process folder of videos/audio files
python highlight_generator.py --input-folder /path/to/videos/

# Transcribe only (useful for audio files)
python highlight_generator.py --input-file audio.mp3 --transcribe-only

# Force overwrite existing transcript files (.json and .txt)
python highlight_generator.py --input-file video.mp4 --transcribe-only --force-overwrite

# Full video processing (skip segment selection)
python highlight_generator.py --input-file video.mp4 --full-video

# Advanced options
python highlight_generator.py --input-file video.mp4 \
    --output custom_output.mp4 \
    --max-duration 180 \
    --zoom-factor 2.5 \
    --vertical-position 0.67 \
    --add-subtitles \
    --manual-title "Custom Title" \
    --skip-logo \
    --skip-background \
    --skip-service-info
```

### Resume Processing
```bash
# Resume from different stages using saved state
python highlight_generator.py --input-file video.mp4 --resume-from transcribe
python highlight_generator.py --input-file video.mp4 --resume-from segments  
python highlight_generator.py --input-file video.mp4 --resume-from titles
```

### Manual Segment Processing
```bash
# Create highlights from specific timestamps (bypasses AI analysis)
python create_segment_highlights.py video.mp4 "10-30,45-75,120-180"

# With custom titles
python create_segment_highlights.py video.mp4 "10:30:Introduction;45:75:Main Point;120:180:Conclusion"

# From JSON file
python create_segment_highlights.py video.mp4 segments.json

# Advanced options with custom output
python create_segment_highlights.py video.mp4 "10-30,45-75" \
    --output-dir ./highlights \
    --output-template "highlight_{index}_{title}_{timestamp}.mp4" \
    --zoom-factor 2.5 \
    --vertical-position 0.67 \
    --skip-logo \
    --skip-background \
    --skip-service-info

# Create JSON template for segments
python create_segment_highlights.py --create-template --template-output my_segments.json
```

## Architecture Overview

### Core Processing Pipeline
1. **Transcription** (`src/transcription/`) - AssemblyAI integration with speaker diarization
2. **Highlight Analysis** (`src/highlights/`) - AI-powered segment identification and title generation
3. **Video Processing** (`src/video/`) - Face tracking, zooming, and layout rendering
4. **Layout Rendering** (`src/layout/`) - 9:16 format composition with titles and subtitles

### Key Components

**Data Models** (`src/models/`):
- `TranscriptionSegment` - Timestamped text with speaker info and word-level data
- `HighlightClip` - Potential highlight with timing, score, and generated title
- `FontManager` - Traditional Chinese font handling

**Video Processing** (`src/video/processor.py`):
- Face detection using MediaPipe
- Smart zoom and tracking with configurable vertical positioning
- Smooth motion with frame-level processing

**Layout System** (`src/layout/`):
- Preview generation for user confirmation
- 9:16 portrait format with 50% video section
- Church service styling with logos, titles, and service info
- Subtitle rendering with smart phrase segmentation

### State Management
The application saves processing state at each major step:
- `{filename}_transcript.state.json` - Transcription results
- `{filename}_segments.state.json` - Identified highlight segments
- `{filename}_titles.state.json` - Generated titles and scores
- `{filename}.transcript.json` - Full transcript with speaker data
- `{filename}.transcript.txt` - Human-readable transcript

### Configuration
- Environment variables in `.env` file for API keys
- Default assets in `default_assets/` directory
- Traditional Chinese fonts in `fonts/` directory
- TensorFlow Lite models in `models/` directory

## API Dependencies

Requires API keys for:
- `ASSEMBLY_API_KEY` - AssemblyAI for transcription with speaker diarization
- `OPENAI_API_KEY` - OpenAI for content analysis and title generation
- `ANTHROPIC_API_KEY` - Claude for highlight identification (currently using OpenAI)

## File Processing

**Supported Input Formats**:
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`

**Output**: 9:16 portrait MP4 optimized for social media with optional Traditional Chinese subtitles.

## Development Notes

- The codebase is designed for Traditional Chinese content but can work with other languages
- Face detection and tracking is optimized for speaking scenarios (church services, presentations)
- The vertical positioning system allows subjects to occupy 2/3 of the frame height by default
- Subtitle segmentation uses jieba for natural Chinese phrase boundaries
- Processing is stateful and resumable at multiple stages for long videos