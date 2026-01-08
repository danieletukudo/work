import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def save_azure_links(participant_identity: str, video_url: str, transcript_url: str, auto_evaluate: bool = True):
    os.makedirs("azure_links", exist_ok=True)
    print(f"Saving Azure links for {participant_identity}")
    print(f"Video URL: {video_url}")
    print(f"Transcript URL: {transcript_url}")





