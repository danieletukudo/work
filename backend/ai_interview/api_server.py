"""
FastAPI server for LiveKit connection details and evaluation management.
Converted from Flask for better performance and async support.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from livekit import api
from dotenv import load_dotenv
from datetime import datetime
import os
import uuid
import json
import glob
from evaluation_formatter import format_evaluation_as_txt, save_evaluation_txt

# Load environment variables from .env
load_dotenv(".env")

app = FastAPI(title="LiveKit Interview & Proctoring API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get LiveKit credentials from environment
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


# Pydantic models for request/response
class ConnectionRequest(BaseModel):
    room_config: Optional[dict] = None
    jobid: Optional[str] = None
    appid: Optional[str] = None


class EvaluationNotification(BaseModel):
    filename: str


class EvaluateFromLinksRequest(BaseModel):
    links_file: str
    participant_identity: str


class ProctoringRequest(BaseModel):
    video_url: str


@app.post('/api/connection-details')
async def get_connection_details(body: Optional[ConnectionRequest] = None):
    """Generate connection details for LiveKit room"""
    
    # Log received parameters
    if body:
        if body.jobid or body.appid:
            print(f"Received parameters - jobid: {body.jobid}, appid: {body.appid}")

    # Validate credentials
    if not LIVEKIT_URL:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL is not defined")
    if not LIVEKIT_API_KEY:
        raise HTTPException(status_code=500, detail="LIVEKIT_API_KEY is not defined")
    if not LIVEKIT_API_SECRET:
        raise HTTPException(status_code=500, detail="LIVEKIT_API_SECRET is not defined")
    
    try:
        # Parse request body for agent configuration
        agent_name = None
        if body and body.room_config and body.room_config.get('agents'):
            agents = body.room_config['agents']
            if len(agents) > 0:
                agent_name = agents[0].get('agent_name')


        
        # Generate unique room and participant identifiers
        # Include jobid and appid in room name for agent to extract
        room_suffix = uuid.uuid4().hex[:8]
        if body and body.jobid and body.appid:
            # Encode jobid and appid in room name: room_jobid_appid_suffix
            room_name = f"voice_assistant_room_{body.jobid}_{body.appid}_{room_suffix}"
        else:
            room_name = f"voice_assistant_room_{room_suffix}"
        participant_identity = f"voice_assistant_user_{uuid.uuid4().hex[:8]}"
        participant_name = "user"
        
        # Create access token
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(participant_identity) \
            .with_name(participant_name) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True
            ))
        
        jwt_token = token.to_jwt()
        
        # Return connection details in the format expected by the frontend
        response = {
            "serverUrl": LIVEKIT_URL,
            "roomName": room_name,
            "participantToken": jwt_token,
            "participantName": participant_name,
        }
        
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "livekit_url_configured": bool(LIVEKIT_URL),
        "api_key_configured": bool(LIVEKIT_API_KEY),
        "api_secret_configured": bool(LIVEKIT_API_SECRET),
    }


@app.post('/api/proctoring/analyze')
async def analyze_video_proctoring(request: ProctoringRequest):
    """Analyze a video for proctoring violations"""
    try:
        video_url = request.video_url
        
        if not video_url:
            raise HTTPException(status_code=400, detail="video_url is required")
        
        
        # Import necessary modules
        from azure_storage import download_video_from_azure
        from video_proctoring_analyzer import analyze_video_proctoring
        
        # Create proctoring_reports directory
        reports_dir = "proctoring_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Download video from Azure
        download_result = download_video_from_azure(video_url, download_folder="proctoring_videos")
        
        if not download_result.get('success'):
            error_msg = download_result.get('error', 'Failed to download video')
            raise HTTPException(status_code=500, detail=error_msg)
        
        video_path = download_result.get('local_file_path')
        
        # Generate report filename
        video_filename = os.path.basename(video_path)
        report_filename = video_filename.replace('.mp4', '_proctoring_report.txt')
        report_path = os.path.join(reports_dir, report_filename)
        
        # Analyze video
        analysis_result = analyze_video_proctoring(video_path, report_path)
        
        if not analysis_result.get('success'):
            error_msg = analysis_result.get('error', 'Analysis failed')
            raise HTTPException(status_code=500, detail=error_msg)
        
        
        # Return results
        return {
            "success": True,
            "report_text": analysis_result.get('report_text'),
            "report_file": report_path,
            "video_url": video_url,
            "video_path": video_path,
            "integrity_score": analysis_result.get('integrity_score'),
            "left_gaze_duration": analysis_result.get('left_gaze_duration'),
            "right_gaze_duration": analysis_result.get('right_gaze_duration'),
            "multiple_face_periods": analysis_result.get('multiple_face_periods'),
            "warnings": analysis_result.get('warnings', []),
            "duration": analysis_result.get('duration')
        }
        
    except HTTPException:
        raise


@app.get('/api/proctoring/reports')
async def list_proctoring_reports():
    """List all proctoring reports"""
    try:
        reports_dir = "proctoring_reports"
        if not os.path.exists(reports_dir):
            return {"reports": []}
        
        # Find all report TXT files
        pattern = os.path.join(reports_dir, "*_proctoring_report.txt")
        report_files = glob.glob(pattern)
        
        reports = []
        for file_path in sorted(report_files, reverse=True):
            try:
                filename = os.path.basename(file_path)
                file_stat = os.stat(file_path)
                
                # Extract video name from report filename
                video_name = filename.replace('_proctoring_report.txt', '.mp4')
                
                reports.append({
                    "filename": filename,
                    "video_name": video_name,
                    "created_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "file_size": file_stat.st_size
                })
            except Exception as e:
                continue
        
        return {"reports": reports}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/proctoring/reports/{filename}')
async def get_proctoring_report(filename: str):
    """Download a proctoring report"""
    try:
        # Security: Only allow report files
        if not filename.endswith('_proctoring_report.txt'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join("proctoring_reports", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            file_path,
            media_type='text/plain',
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/proxy-video')
async def proxy_video(request: Request, url: str):
    """Proxy Azure video with proper streaming support for browser playback"""
    try:
        if not url:
            raise HTTPException(status_code=400, detail="url parameter is required")
        
        # Download video from Azure to local cache
        from azure_storage import download_video_from_azure
        
        # Use cache directory
        cache_dir = "video_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        
        download_result = download_video_from_azure(url, download_folder=cache_dir)
        
        if not download_result.get('success'):
            error_msg = download_result.get('error', 'Failed to download video')
            raise HTTPException(status_code=500, detail=error_msg)
        
        video_path = download_result.get('local_file_path')
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=500, detail="Video file not found after download")
        
        
        # Get file size for range requests
        file_size = os.path.getsize(video_path)
        
        # Check if this is a range request
        range_header = request.headers.get('range', None)
        
        if range_header:
            # Parse range header (e.g., "bytes=0-1023")
            byte_start = 0
            byte_end = file_size - 1
            
            try:
                range_match = range_header.replace('bytes=', '').split('-')
                if range_match[0]:
                    byte_start = int(range_match[0])
                if len(range_match) > 1 and range_match[1]:
                    byte_end = int(range_match[1])
            except:
                pass
            
            # Ensure byte_end doesn't exceed file size
            byte_end = min(byte_end, file_size - 1)
            
            # Read the requested chunk
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                chunk_size = byte_end - byte_start + 1
                data = f.read(chunk_size)
            
            # Return 206 Partial Content
            headers = {
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(len(data)),
                'Content-Type': 'video/mp4',
                'Cache-Control': 'public, max-age=3600',
            }
            
            return Response(content=data, status_code=206, headers=headers, media_type='video/mp4')
        else:
            # No range request - send full file
            return FileResponse(
                video_path,
                media_type='video/mp4',
                headers={
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'public, max-age=3600'
                }
            )
    
    except HTTPException:
        raise


if __name__ == '__main__':
    import uvicorn
    
    print("=" * 60)
    print("🚀 LiveKit Connection API Server (FastAPI)")
    print("=" * 60)
    print(f"📍 LIVEKIT_URL: {LIVEKIT_URL[:30]}..." if LIVEKIT_URL else "❌ LIVEKIT_URL not set")
    print(f"🔑 API Key configured: {bool(LIVEKIT_API_KEY)}")
    print(f"🔐 API Secret configured: {bool(LIVEKIT_API_SECRET)}")
    print("=" * 60)
    print("🌐 Starting server on http://localhost:5001")
    print("=" * 60)
    
    uvicorn.run(app, host='0.0.0.0', port=5001, log_level="info")

