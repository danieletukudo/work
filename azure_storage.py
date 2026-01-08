"""
Azure Blob Storage Module for Video Upload/Download
Handles video file uploads to and downloads from Azure Blob Storage
"""

import os
import logging
from datetime import datetime
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")

# Azure Storage Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_ACCESS_KEY = os.getenv("AZURE_STORAGE_ACCESS_KEY")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "uploads")
AZURE_STORAGE_URL = os.getenv("AZURE_STORAGE_URL")
SAS_TOKEN = os.getenv("SAS_TOKEN")


class AzureVideoStorage:
    """Handle Azure Blob Storage operations for video files"""
    
    def __init__(self):
        """Initialize Azure Blob Storage client"""
        if not AZURE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            self.container_name = BLOB_CONTAINER
        except Exception as e:
            raise
    
    def upload_video(
        self, 
        local_file_path: str, 
        blob_name: Optional[str] = None,
        subfolder: str = "videos"
    ) -> dict:
        """
        Upload video file to Azure Blob Storage
        
        Args:
            local_file_path (str): Path to the local video file
            blob_name (str, optional): Name for the blob in Azure. If None, uses filename
            subfolder (str): Subfolder within the container (default: "videos")
        
        Returns:
            dict: Upload result with status, blob_url, and blob_name
        
        Example:
            storage = AzureVideoStorage()
            result = storage.upload_video("recordings/user123_20250104.mp4")
        """
        try:
            # Check if file exists
            if not os.path.exists(local_file_path):
                error_msg = f"File not found: {local_file_path}"
                return {
                    "success": False,
                    "error": error_msg,
                    "blob_url": None,
                    "blob_name": None
                }
            
            # Get file size
            file_size = os.path.getsize(local_file_path)
            
            # Generate blob name if not provided
            if blob_name is None:
                blob_name = os.path.basename(local_file_path)
            
            # Add subfolder to blob name
            if subfolder:
                blob_name = f"{subfolder}/{blob_name}"
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Upload file
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Generate blob URL
            if SAS_TOKEN:
                blob_url = f"{AZURE_STORAGE_URL}/{self.container_name}/{blob_name}{SAS_TOKEN}"
            else:
                blob_url = f"{AZURE_STORAGE_URL}/{self.container_name}/{blob_name}"
            
            
            return {
                "success": True,
                "blob_url": blob_url,
                "blob_name": blob_name,
                "file_size": file_size,
                "container": self.container_name
            }
            
        except Exception as e:
            error_msg = f"Failed to upload video: {str(e)}"
            return {
                "success": False,
                "error": error_msg,
                "blob_url": None,
                "blob_name": None
            }
    
    def download_video(
        self, 
        blob_name_or_url: str, 
        local_file_path: Optional[str] = None,
        download_folder: str = "downloads"
    ) -> dict:
        """
        Download video file from Azure Blob Storage
        
        Args:
            blob_name_or_url (str): Blob name (e.g., "videos/video.mp4") or full Azure URL
            local_file_path (str, optional): Local path to save the file. 
                                            If None, saves to download_folder
            download_folder (str): Folder to save downloads (default: "downloads")
        
        Returns:
            dict: Download result with status and local_file_path
        
        Example:
            storage = AzureVideoStorage()
            # Using blob name
            result = storage.download_video("videos/user123_20250104.mp4")
            # Using full URL
            result = storage.download_video("https://storage.azure.com/container/videos/video.mp4?sv=...")
        """
        try:
            # Extract blob name from URL if full URL is provided
            blob_name = blob_name_or_url
            if blob_name_or_url.startswith("http://") or blob_name_or_url.startswith("https://"):
                # Parse the URL to extract blob name
                parsed_url = urlparse(blob_name_or_url)
                # Path format: /container_name/subfolder/filename
                path_parts = parsed_url.path.strip('/').split('/', 1)
                if len(path_parts) > 1:
                    # Remove container name, keep subfolder/filename
                    blob_name = path_parts[1]
                else:
                    blob_name = path_parts[0]
            
            # Generate local file path if not provided
            if local_file_path is None:
                os.makedirs(download_folder, exist_ok=True)
                filename = os.path.basename(blob_name)
                local_file_path = os.path.join(download_folder, filename)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Check if blob exists
            if not blob_client.exists():
                error_msg = f"Blob not found: {blob_name}"
                return {
                    "success": False,
                    "error": error_msg,
                    "local_file_path": None
                }
            
            # Download blob
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            file_size = os.path.getsize(local_file_path)
            
            return {
                "success": True,
                "local_file_path": local_file_path,
                "blob_name": blob_name,
                "file_size": file_size
            }
            
        except Exception as e:
            error_msg = f"Failed to download video: {str(e)}"
            return {
                "success": False,
                "error": error_msg,
                "local_file_path": None
            }
    
    def list_videos(self, subfolder: str = "videos") -> list:
        """
        List all videos in a specific subfolder
        
        Args:
            subfolder (str): Subfolder to list videos from (default: "videos")
        
        Returns:
            list: List of blob names
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            blobs = []
            for blob in container_client.list_blobs(name_starts_with=subfolder):
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "created": blob.creation_time,
                    "url": f"{AZURE_STORAGE_URL}/{self.container_name}/{blob.name}"
                })
            
            return blobs
            
        except Exception as e:
            return []
    
    def delete_video(self, blob_name: str) -> dict:
        """
        Delete video from Azure Blob Storage
        
        Args:
            blob_name (str): Name of the blob to delete
        
        Returns:
            dict: Deletion result with status
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            if not blob_client.exists():
                return {
                    "success": False,
                    "error": f"Blob not found: {blob_name}"
                }
            
            blob_client.delete_blob()
            
            return {
                "success": True,
                "blob_name": blob_name
            }
            
        except Exception as e:
            error_msg = f"Failed to delete video: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_video_url(self, blob_name: str, with_sas: bool = True) -> str:
        """
        Get the URL for a video blob
        
        Args:
            blob_name (str): Name of the blob
            with_sas (bool): Include SAS token in URL (default: True)
        
        Returns:
            str: Full URL to the blob
        """
        base_url = f"{AZURE_STORAGE_URL}/{self.container_name}/{blob_name}"
        if with_sas and SAS_TOKEN:
            return f"{base_url}{SAS_TOKEN}"
        return base_url
    
    def upload_transcript(
        self,
        local_file_path: str,
        blob_name: Optional[str] = None,
        subfolder: str = "transcripts"
    ) -> dict:
        """
        Upload transcript JSON file to Azure Blob Storage
        
        Args:
            local_file_path (str): Path to the local transcript JSON file
            blob_name (str, optional): Name for the blob in Azure. If None, uses filename
            subfolder (str): Subfolder within the container (default: "transcripts")
        
        Returns:
            dict: Upload result with status, blob_url, and blob_name
        """
        # Reuse the upload_video method but with transcripts subfolder
        return self.upload_video(local_file_path, blob_name, subfolder)


# Convenience functions for easy use
def upload_video_to_azure(local_file_path: str, blob_name: Optional[str] = None) -> dict:
    """
    Quick function to upload a video to Azure
    
    Args:
        local_file_path (str): Path to local video file
        blob_name (str, optional): Custom name for the blob
    
    Returns:
        dict: Upload result
    
    Example:
        result = upload_video_to_azure("recordings/user123_20250104.mp4")
        if result['success']:
    """
    storage = AzureVideoStorage()
    return storage.upload_video(local_file_path, blob_name)


def download_video_from_azure(blob_name_or_url: str, local_file_path: Optional[str] = None, download_folder: str = "downloads") -> dict:
    """
    Quick function to download a video from Azure using blob name or full URL
    
    Args:
        blob_name_or_url (str): Blob name (e.g., "videos/video.mp4") or full Azure URL
        local_file_path (str, optional): Where to save the file. If None, saves to download_folder
        download_folder (str): Folder to save downloads if local_file_path is not provided (default: "downloads")
    
    Returns:
        dict: Download result
    
    Example:
        # Using blob name
        result = download_video_from_azure("videos/user123_20250104.mp4")
        # Using full URL from saved links
        result = download_video_from_azure("https://storage.azure.com/container/videos/video.mp4?sv=...")
        # With custom download folder
        result = download_video_from_azure("videos/video.mp4", download_folder="my_downloads")
        if result['success']:
    """
    storage = AzureVideoStorage()
    return storage.download_video(blob_name_or_url, local_file_path, download_folder)


def upload_transcript_to_azure(local_file_path: str, blob_name: Optional[str] = None) -> dict:
    """
    Quick function to upload a transcript to Azure
    
    Args:
        local_file_path (str): Path to local transcript JSON file
        blob_name (str, optional): Custom name for the blob
    
    Returns:
        dict: Upload result
    
    Example:
        result = upload_transcript_to_azure("transcript_20250104.json")
        if result['success']:
    """
    storage = AzureVideoStorage()
    return storage.upload_transcript(local_file_path, blob_name)


def download_from_link_file(links_file_path: str, download_video: bool = True, 
                            download_transcript: bool = True, download_folder: str = "downloads") -> dict:
    """
    Download video and/or transcript from a saved links JSON file
    
    Args:
        links_file_path (str): Path to the links JSON file (e.g., "azure_links/user_20250124_links.json")
        download_video (bool): Whether to download the video (default: True)
        download_transcript (bool): Whether to download the transcript (default: True)
        download_folder (str): Folder to save downloads (default: "downloads")
    
    Returns:
        dict: Download results with status for each file
    
    Example:
        result = download_from_link_file("azure_links/user_20250124_links.json")
        if result['video']['success']:
        if result['transcript']['success']:
    """
    import json
    
    try:
        # Read the links file
        with open(links_file_path, 'r') as f:
            links_data = json.load(f)
        
        video_url = links_data.get('video_url')
        transcript_url = links_data.get('transcript_url')
        
        results = {}
        
        # Download video if requested and URL exists
        if download_video and video_url:
            results['video'] = download_video_from_azure(video_url, download_folder=download_folder)
        else:
            results['video'] = {"success": False, "error": "Video URL not found or download skipped"}
        
        # Download transcript if requested and URL exists
        if download_transcript and transcript_url:
            results['transcript'] = download_video_from_azure(transcript_url, download_folder=download_folder)
        else:
            results['transcript'] = {"success": False, "error": "Transcript URL not found or download skipped"}
        
        return results
        
    except FileNotFoundError:
        error_msg = f"Links file not found: {links_file_path}"
        return {
            "video": {"success": False, "error": error_msg},
            "transcript": {"success": False, "error": error_msg}
        }
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in links file: {e}"
        return {
            "video": {"success": False, "error": error_msg},
            "transcript": {"success": False, "error": error_msg}
        }
    except Exception as e:
        error_msg = f"Failed to download from link file: {e}"
        return {
            "video": {"success": False, "error": error_msg},
            "transcript": {"success": False, "error": error_msg}
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Azure Video Storage Module ===\n")
    
    # Initialize storage
    try:
        storage = AzureVideoStorage()
        # print("Azure Storage client initialized\n")
        
        # # Example 1: Upload a video
        # print("Example 1: Upload Video")
        # print("-" * 50)
        # result = storage.upload_video("/Users/danielsamuel/PycharmProjects/DSA/LiveKit-AI-Car-Call-Centre/recordings/playground-user_20251107_155540.mp4")
        # print(f"Result: {result}\n")
        
        # # # Example 2: List videos
        # print("Example 2: List Videos")
        # print("-" * 50)
        # # videos = storage.list_videos(subfolder="videos")
        # for video in videos[:5]:  # Show first 5
        #     print(f"- {video['name']} ({video['size'] / (1024*1024):.2f} MB)")
        # print()
        
        # Example 3: Download a video
        # print("Example 3: Download Video")
        # print("-" * 50)
        result = storage.download_video("https://remotingwork.blob.core.windows.net/uploads/videos/voice_assistant_user_60988ac9_20260108_133022_COMBINED.mp4")
    except Exception as e:
        print(f"Error: {e}")

