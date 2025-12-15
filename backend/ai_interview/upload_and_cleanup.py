"""
Module for uploading video/transcript to Azure and cleaning up local files.

This module handles:
1. Uploading transcript and video files to Azure Blob Storage
2. Saving Azure links to a JSON file
3. Triggering evaluation via API
4. Cleaning up local files after successful upload
"""

import os
import time
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def upload_and_save_links_background(
    participant_identity: str,
    transcript_path: str,
    combined_video_path: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    evaluation_results: Optional[dict] = None
):
    """
    Upload video to Azure, wait for evaluation, then print both results together.
    
    This function runs in a background thread (daemon) so it won't block the main process.
    """
    try:
        time.sleep(0.5)  # Brief delay to ensure files are fully written

        from azure_storage import upload_video_to_azure

        # Upload video if ready
        video_url = None
        if combined_video_path and os.path.exists(combined_video_path):
            video_result = upload_video_to_azure(combined_video_path)

            if video_result.get('success'):
                video_url = video_result['blob_url']

        # Wait for evaluation result (with timeout)
        evaluation_result = None
        if evaluation_results:
            max_wait = 60  # Wait up to 60 seconds for evaluation
            waited = 0
            while waited < max_wait:
                if participant_identity in evaluation_results:
                    evaluation_result = evaluation_results[participant_identity]
                    break
                time.sleep(1)
                waited += 1

        # Print both results together
        if video_url:
            print("result {")
            print(f"  {video_url}")
            if evaluation_result:
                if isinstance(evaluation_result, tuple):
                    eval_str = " ".join(str(x) for x in evaluation_result)
                    print(f"  {eval_str}")
                else:
                    print(f"  {evaluation_result}")
            else:
                print("  evaluation pending")
            print("}")
            
            try:
                if combined_video_path and os.path.exists(combined_video_path):
                    os.remove(combined_video_path)
                if os.path.exists(transcript_path):
                    os.remove(transcript_path)
            except Exception:
                pass

    except Exception:
        pass


def start_upload_in_background(
    participant_identity: str,
    transcript_path: str,
    combined_video_path: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    evaluation_results: Optional[dict] = None
) -> threading.Thread:
    """
    Start the upload and cleanup process in a background daemon thread.
    
    Args:
        participant_identity: The identity of the participant
        transcript_path: Path to the transcript JSON file
        combined_video_path: Optional path to the combined video file
        video_path: Optional path to the original video file
        audio_path: Optional path to the original audio file
    
    Returns:
        The thread object (daemon thread, won't block exit)
    """
    upload_thread = threading.Thread(
        target=upload_and_save_links_background,
        args=(participant_identity, transcript_path, combined_video_path, video_path, audio_path, evaluation_results),
        daemon=True,
        name=f"UploadThread-{participant_identity}"
    )
    upload_thread.start()
    return upload_thread

