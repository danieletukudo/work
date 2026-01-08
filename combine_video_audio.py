"""
Module for combining video and audio files into a single MP4 file.

This module handles the combination of separate video and audio recordings
into a synchronized MP4 file using ffmpeg.
"""

import os
import subprocess
from datetime import datetime
from typing import Optional, Dict


async def combine_video_audio(
        participant_identity: str,
        participant_recording_files: Dict[str, Dict[str, str]]
) -> Optional[str]:
    """
    Combine video and audio files into a single MP4 with perfect sync.

    Args:
        participant_identity: The identity of the participant
        participant_recording_files: Dictionary tracking recording files for each participant
            Format: {participant_identity: {'video': path, 'audio': path}}

    Returns:
        Path to the combined video file, or None if combination failed
    """
    if participant_identity not in participant_recording_files:
        return None

    files = participant_recording_files[participant_identity]
    video_file = files.get('video')
    audio_file = files.get('audio')

    if not video_file or not audio_file:
        return None

    if not os.path.exists(video_file) or not os.path.exists(audio_file):
        return None

    os.makedirs("recordings/combined", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = f"recordings/combined/{participant_identity}_{timestamp}_COMBINED.mp4"

    try:
        # Use ffmpeg to combine video and audio
        # -c:v libx264 = H.264 codec (browser-compatible)
        # -preset ultrafast = fastest encoding (for real-time)
        # -c:a aac = encode audio as AAC
        # -shortest = finish when shortest stream ends
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-i', audio_file,
            '-c:v', 'libx264',  # H.264 for browser compatibility
            '-preset', 'ultrafast',  # Fast encoding
            '-c:a', 'aac',
            '-shortest',
            '-y',  # Overwrite output file if exists
            combined_file
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            if participant_identity not in participant_recording_files:
                participant_recording_files[participant_identity] = {}
            participant_recording_files[participant_identity]['combined'] = combined_file
            return combined_file
        else:
            return None

    except FileNotFoundError:
        return None
    except Exception:
        return None
