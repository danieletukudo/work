"""
Flask API server for LiveKit connection details.
This replaces the Next.js API route so we can use the Python environment.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Get LiveKit credentials from environment
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


@app.route('/api/connection-details', methods=['POST'])
def get_connection_details():
    """Generate connection details for LiveKit room"""
    
    # Validate credentials
    if not LIVEKIT_URL:
        return jsonify({"error": "LIVEKIT_URL is not defined"}), 500
    if not LIVEKIT_API_KEY:
        return jsonify({"error": "LIVEKIT_API_KEY is not defined"}), 500
    if not LIVEKIT_API_SECRET:
        return jsonify({"error": "LIVEKIT_API_SECRET is not defined"}), 500
    
    try:
        # Parse request body for agent configuration
        body = request.get_json() or {}
        agent_name = None
        if body.get('room_config') and body['room_config'].get('agents'):
            agents = body['room_config']['agents']
            if len(agents) > 0:
                agent_name = agents[0].get('agent_name')
        
        # Generate unique room and participant identifiers
        room_name = f"voice_assistant_room_{uuid.uuid4().hex[:8]}"
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
        
        print(f" Generated connection details for room: {room_name}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f" Error generating connection details: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "livekit_url_configured": bool(LIVEKIT_URL),
        "api_key_configured": bool(LIVEKIT_API_KEY),
        "api_secret_configured": bool(LIVEKIT_API_SECRET),
    })


@app.route('/api/evaluations', methods=['GET'])
def list_evaluations():
    """List all available evaluation files"""
    try:
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            return jsonify({"evaluations": []})
        
        # Find all evaluation JSON files
        pattern = os.path.join(downloads_dir, "evaluation_*.json")
        evaluation_files = glob.glob(pattern)
        
        evaluations = []
        for file_path in sorted(evaluation_files, reverse=True):  # Most recent first
            try:
                filename = os.path.basename(file_path)
                file_stat = os.stat(file_path)
                
                # Read evaluation to get metadata
                with open(file_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                evaluations.append({
                    "filename": filename,
                    "candidate_id": eval_data.get('candidate_id', 'unknown'),
                    "candidate_name": eval_data.get('candidate_name', 'Unknown'),
                    "job_id": eval_data.get('job_id', 'unknown'),
                    "overall_score": eval_data.get('overall_score', 0),
                    "hiring_recommendation": eval_data.get('hiring_recommendation', 'N/A'),
                    "created_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "file_size": file_stat.st_size
                })
            except Exception as e:
                print(f"Error reading evaluation file {file_path}: {e}")
                continue
        
        return jsonify({"evaluations": evaluations})
    
    except Exception as e:
        print(f"Error listing evaluations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/<filename>', methods=['GET'])
def get_evaluation(filename: str):
    """Download evaluation as JSON"""
    try:
        # Security: Only allow evaluation files
        if not filename.startswith('evaluation_') or not filename.endswith('.json'):
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join("downloads", filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Evaluation not found"}), 404
        
        return send_file(
            file_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Error downloading evaluation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/<filename>/txt', methods=['GET'])
def get_evaluation_txt(filename: str):
    """Download evaluation as TXT"""
    try:
        # Security: Only allow evaluation files
        if not filename.startswith('evaluation_') or not filename.endswith('.json'):
            return jsonify({"error": "Invalid filename"}), 400
        
        json_path = os.path.join("downloads", filename)
        
        if not os.path.exists(json_path):
            return jsonify({"error": "Evaluation not found"}), 404
        
        # Load evaluation JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            evaluation = json.load(f)
        
        # Convert to TXT
        txt_content = format_evaluation_as_txt(evaluation)
        
        # Create temporary TXT file
        txt_filename = filename.replace('.json', '.txt')
        txt_path = os.path.join("downloads", txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        return send_file(
            txt_path,
            mimetype='text/plain',
            as_attachment=True,
            download_name=txt_filename
        )
    
    except Exception as e:
        print(f"Error generating TXT evaluation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/latest', methods=['GET'])
def get_latest_evaluation():
    """Get the most recent evaluation"""
    try:
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            return jsonify({"error": "No evaluations found"}), 404
        
        pattern = os.path.join(downloads_dir, "evaluation_*.json")
        evaluation_files = glob.glob(pattern)
        
        if not evaluation_files:
            return jsonify({"error": "No evaluations found"}), 404
        
        # Get most recent file
        latest_file = max(evaluation_files, key=os.path.getmtime)
        filename = os.path.basename(latest_file)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            evaluation = json.load(f)
        
        # Check if TXT file exists
        txt_filename = filename.replace('.json', '.txt')
        txt_path = os.path.join(downloads_dir, txt_filename)
        txt_exists = os.path.exists(txt_path)
        
        return jsonify({
            "filename": filename,
            "evaluation": evaluation,
            "txt_available": txt_exists,
            "txt_filename": txt_filename if txt_exists else None
        })
    
    except Exception as e:
        print(f"Error getting latest evaluation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/latest/txt', methods=['GET'])
def get_latest_evaluation_txt():
    """Get the most recent evaluation as TXT"""
    try:
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            return jsonify({"error": "No evaluations found"}), 404
        
        pattern = os.path.join(downloads_dir, "evaluation_*.txt")
        txt_files = glob.glob(pattern)
        
        if not txt_files:
            return jsonify({"error": "No evaluation TXT files found"}), 404
        
        # Get most recent file
        latest_file = max(txt_files, key=os.path.getmtime)
        filename = os.path.basename(latest_file)
        
        return send_file(
            latest_file,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Error getting latest evaluation TXT: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/status', methods=['GET'])
def get_evaluation_status():
    """Get evaluation status - check if there's a new evaluation or one in progress"""
    try:
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            return jsonify({
                "status": "no_evaluations",
                "latest_evaluation": None,
                "evaluation_count": 0
            })
        
        pattern = os.path.join(downloads_dir, "evaluation_*.json")
        evaluation_files = glob.glob(pattern)
        
        if not evaluation_files:
            return jsonify({
                "status": "no_evaluations",
                "latest_evaluation": None,
                "evaluation_count": 0
            })
        
        # Get most recent file
        latest_file = max(evaluation_files, key=os.path.getmtime)
        filename = os.path.basename(latest_file)
        file_mtime = os.path.getmtime(latest_file)
        
        # Check if file was modified in the last 5 minutes (evaluation might be in progress)
        current_time = datetime.now().timestamp()
        time_diff = current_time - file_mtime
        
        # Get file size to check if it's being written
        file_size = os.path.getsize(latest_file)
        
        return jsonify({
            "status": "ready",
            "latest_evaluation": {
                "filename": filename,
                "modified_time": file_mtime,
                "modified_ago_seconds": time_diff,
                "file_size": file_size,
                "is_recent": time_diff < 300  # Less than 5 minutes old
            },
            "evaluation_count": len(evaluation_files)
        })
    
    except Exception as e:
        print(f"Error getting evaluation status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/evaluations/notify_new', methods=['POST'])
def notify_new_evaluation():
    """Receive notification from agent when a new evaluation is ready"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        print(f" Received notification for new evaluation: {filename}")
        # Here you could add logic to push updates to connected clients (e.g., via websockets)
        return jsonify({"status": "success", "message": "Notification received"}), 200
    except Exception as e:
        print(f" Error processing new evaluation notification: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/evaluations/evaluate_from_links', methods=['POST'])
def evaluate_from_links():
    """
    Evaluate interview from Azure links file (runs independently from agent)
    This endpoint downloads transcript from Azure and runs evaluation
    """
    try:
        data = request.get_json()
        links_file = data.get('links_file')
        participant_identity = data.get('participant_identity')
        
        if not links_file:
            return jsonify({"error": "links_file is required"}), 400
        
        print(f" Starting evaluation from links file: {links_file}")
        
        # Import evaluation function
        from save_links import download_and_evaluate_from_links
        import threading
        
        # Run evaluation in background thread (daemon so API can respond immediately)
        def run_evaluation_background():
            """Run evaluation in background"""
            try:
                print(f" Downloading and evaluating from: {links_file}")
                
                result = download_and_evaluate_from_links(
                    links_file_path=links_file,
                    job_description=None,
                    candidate_cv=None,
                    candidate_info=None,
                    evaluation_instruction=None,
                    use_api=False
                )
                
                if result and result.get('success'):
                    print(f" Evaluation completed successfully!")
                    print(f" Score: {result.get('evaluation', {}).get('overall_score', 'N/A')}/10")
                    
                    # Notify frontend
                    try:
                        import requests
                        api_url = os.getenv('API_SERVER_URL', 'http://localhost:5001')
                        eval_file = result.get('evaluation_file_absolute') or result.get('evaluation_file')
                        if eval_file:
                            requests.post(f"{api_url}/api/evaluations/notify_new",
                                         json={"filename": os.path.basename(eval_file)}, timeout=2)
                            print(f" Frontend notified")
                    except:
                        pass
                else:
                    print(f" Evaluation failed: {result.get('error') if result else 'No result'}")
                    
            except Exception as e:
                print(f" Error in background evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        # Start evaluation in daemon thread (won't block API response)
        eval_thread = threading.Thread(target=run_evaluation_background, daemon=True)
        eval_thread.start()
        
        print(f" Evaluation started in background")
        
        # Return immediately (evaluation runs in background)
        return jsonify({
            "status": "success",
            "message": "Evaluation started in background",
            "links_file": links_file
        }), 202  # 202 Accepted (processing in background)
        
    except Exception as e:
        error_msg = f"Error starting evaluation: {str(e)}"
        print(f" {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/proctoring/analyze', methods=['POST'])
def analyze_video_proctoring():
    """Analyze a video for proctoring violations"""
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({"error": "video_url is required"}), 400
        
        print(f" Starting proctoring analysis for: {video_url}")
        
        # Import necessary modules
        from azure_storage import download_video_from_azure
        from video_proctoring_analyzer import analyze_video_proctoring
        
        # Create proctoring_reports directory
        reports_dir = "proctoring_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Download video from Azure
        print(f" Downloading video from Azure...")
        download_result = download_video_from_azure(video_url, download_folder="proctoring_videos")
        
        if not download_result.get('success'):
            error_msg = download_result.get('error', 'Failed to download video')
            print(f" Download failed: {error_msg}")
            return jsonify({"error": error_msg}), 500
        
        video_path = download_result.get('local_file_path')
        print(f" Video downloaded to: {video_path}")
        
        # Generate report filename
        video_filename = os.path.basename(video_path)
        report_filename = video_filename.replace('.mp4', '_proctoring_report.txt')
        report_path = os.path.join(reports_dir, report_filename)
        
        # Analyze video
        print(f" Analyzing video for proctoring violations...")
        analysis_result = analyze_video_proctoring(video_path, report_path)
        
        if not analysis_result.get('success'):
            error_msg = analysis_result.get('error', 'Analysis failed')
            print(f" Analysis failed: {error_msg}")
            return jsonify({"error": error_msg}), 500
        
        print(f" Analysis completed successfully!")
        print(f" Integrity Score: {analysis_result.get('integrity_score', 0):.2f}%")
        
        # Return results
        return jsonify({
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
        }), 200
        
    except Exception as e:
        error_msg = f"Error in proctoring analysis: {str(e)}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


@app.route('/api/proctoring/reports', methods=['GET'])
def list_proctoring_reports():
    """List all proctoring reports"""
    try:
        reports_dir = "proctoring_reports"
        if not os.path.exists(reports_dir):
            return jsonify({"reports": []})
        
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
                print(f"Error reading report file {file_path}: {e}")
                continue
        
        return jsonify({"reports": reports})
    
    except Exception as e:
        print(f"Error listing reports: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/proctoring/reports/<filename>', methods=['GET'])
def get_proctoring_report(filename: str):
    """Download a proctoring report"""
    try:
        # Security: Only allow report files
        if not filename.endswith('_proctoring_report.txt'):
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join("proctoring_reports", filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Report not found"}), 404
        
        return send_file(
            file_path,
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Error downloading report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/proxy-video', methods=['GET'])
def proxy_video():
    """Proxy Azure video with proper streaming support for browser playback"""
    try:
        video_url = request.args.get('url')
        
        if not video_url:
            return jsonify({"error": "url parameter is required"}), 400
        
        # Download video from Azure to local cache
        from azure_storage import download_video_from_azure
        
        # Use cache directory
        cache_dir = "video_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f" Proxying video from: {video_url[:80]}...")
        
        download_result = download_video_from_azure(video_url, download_folder=cache_dir)
        
        if not download_result.get('success'):
            error_msg = download_result.get('error', 'Failed to download video')
            print(f" Download failed: {error_msg}")
            return jsonify({"error": error_msg}), 500
        
        video_path = download_result.get('local_file_path')
        
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not found after download"}), 500
        
        print(f" Video cached: {video_path}")
        
        # Get file size for range requests
        file_size = os.path.getsize(video_path)
        
        # Check if this is a range request
        range_header = request.headers.get('Range', None)
        
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
            
            # Read the requested chunk
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                chunk_size = byte_end - byte_start + 1
                data = f.read(chunk_size)
            
            # Return 206 Partial Content
            response = app.response_class(
                data,
                206,
                mimetype='video/mp4',
                direct_passthrough=True
            )
            response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = str(len(data))
            response.headers['Cache-Control'] = 'public, max-age=3600'
            
            return response
        else:
            # No range request - send full file
            response = send_file(
                video_path,
                mimetype='video/mp4',
                as_attachment=False,
                conditional=True
            )
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
    
    except Exception as e:
        error_msg = f"Error proxying video: {str(e)}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    print("=" * 60)
    print(" LiveKit Connection API Server")
    print("=" * 60)
    print(f" LIVEKIT_URL: {LIVEKIT_URL[:30]}..." if LIVEKIT_URL else " LIVEKIT_URL not set")
    print(f" API Key configured: {bool(LIVEKIT_API_KEY)}")
    print(f" API Secret configured: {bool(LIVEKIT_API_SECRET)}")
    print("=" * 60)
    print(" Starting server on http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)

