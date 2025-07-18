import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import whisper
from moviepy import *
from open_clip import create_model_and_transforms


def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Extract audio from video and save as WAV file.

    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to save the audio file.

    Returns:
        str: Path to the saved audio file.
    """
    video_name = Path(video_path).stem
    audio_path = Path(output_dir) / f"{video_name}.wav"

    try:
        video = VideoFileClip(str(video_path))
        audio = video.audio

        if audio is not None:
            audio.write_audiofile(str(audio_path), logger="bar")
            audio.close()
        video.close()

        if not audio_path.exists():
            raise RuntimeError("Audio file was not created")

        return str(audio_path)

    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return ""


def transcribe_audio(audio_path: str, model_name: str = "base") -> str:
    """
    Transcribe audio using Whisper.

    Args:
        audio_path (str): Path to the audio file.
        model_name (str): Whisper model name.

    Returns:
        str: Transcription text.
    """
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path), fp16=False, verbose=False)
        return str(result.get("text", "")).strip()

    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {str(e)}")


def transcribe_video(
    video_path: str,
    output_dir: str = "g3/data/prompt_data/audio",
    model_name: str = "base",
):
    """
    Transcribe video by extracting audio and then transcribing it.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the audio file.
        model_name (str): Whisper model name.

    Returns:
        str: Path to the saved transcription text file.
    """
    audio_path = extract_audio(video_path, output_dir)
    if not audio_path:
        print("Audio extraction failed. No audio file created.")
        return

    print(f"Audio extracted to: {audio_path}")
    transcript_text = transcribe_audio(audio_path, model_name=model_name)

    transcript_path = Path(output_dir) / f"{Path(video_path).stem}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"Transcript saved to: {transcript_path}")


def transcribe_video_directory(
    video_dir: str,
    output_dir: str = "g3/data/prompt_data/audio",
    model_name: str = "base",
):
    """
    Transcribe all videos in a directory.

    Args:
        video_dir (str): Directory containing video files.
        output_dir (str): Directory to save the audio and transcript files.
        model_name (str): Whisper model name.

    Returns:
        None
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [
        f for f in Path(video_dir).glob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in directory: {video_dir}")

    for video_file in video_files:
        print(f"Processing video: {video_file}")
        transcribe_video(str(video_file), output_dir, model_name=model_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing.py <video_path>")
        print("Example: python data_preprocessing.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        sys.exit(1)
    print(f"Processing video: {video_path}")
    print("=" * 60)

    output_dir = "g3/data/prompt_data/audio"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(video_path):
        print(f"📂 Detected directory: {video_path}")
        transcribe_video_directory(video_path, output_dir)
        print("🎉 All videos processed successfully!")
        sys.exit(0)
    else:
        try:
            print("🎵 Extracting audio...")
            audio_path = extract_audio(video_path, output_dir)
            print(f"✅ Audio extracted to: {audio_path}")

            print("📝 Transcribing audio...")
            transcript_text = transcribe_audio(audio_path)

            transcript_path = (
                Path(output_dir) / f"{Path(video_path).stem}_transcript.txt"
            )
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            print(f"✅ Transcript saved to: {transcript_path}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("🎉 Processing completed!")
