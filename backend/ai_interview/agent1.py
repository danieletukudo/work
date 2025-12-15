import sys

from prompt import get_instruction
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
import subprocess
from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    utils,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.agents.voice.room_io import RoomOutputOptions
from dotenv import load_dotenv
import os
from livekit.agents import AgentSession, Agent, RoomInputOptions, inference
from livekit.agents import ModelSettings
from livekit import rtc
from typing import AsyncIterable, Optional
import logging
from livekit import api
from livekit.api.egress_service import EgressService, ParticipantEgressRequest
from google.protobuf import message_factory
import cv2
import numpy as np
import wave
from azure_storage import AzureVideoStorage, upload_video_to_azure, upload_transcript_to_azure
from save_links import save_azure_links
import threading
import re
from call import get_ai_cv_data

#
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(".env")

# Global transcript storage for real-time updates
transcript_data = {
    "user_transcript": "",
    "agent_transcript": "",
    "conversation_history": []
}

# Global storage for interview metadata
interview_metadata = {
    "job_title": None,
    "candidate_name": None
}


def clean_html(text):
    """Remove HTML tags and clean up text"""
    if not isinstance(text, str):
        return text
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = text.replace('\\"', '')       # remove escaped quotes
    text = text.replace('"', '')         # remove extra quotes
    return text.strip()


def fetch_job_and_candidate_data(jobid: Optional[str] = None, appid: Optional[str] = None):
    """
    Fetch job and candidate data from API using jobid and appid.
    jobid and appid are only used here to call call.py's get_ai_cv_data().
    Returns only the fields provided by the API output (no jobid/appid in return).
    """
    # jobid and appid must be provided (extracted from room name or passed in)
    if not jobid or not appid:
        logger.warning(f"⚠️ Missing jobid or appid. jobid: {jobid}, appid: {appid}")
        return None
    
    try:
        logger.info(f"📡 Fetching job and candidate data - jobid: {jobid}, appid: {appid}")
        # Fetch data from API
        api_data = get_ai_cv_data(jobid, appid)
        
        if not api_data or not api_data.get('status'):
            logger.error(f"❌ Failed to fetch data from API: {api_data}")
            return None
        
        data = api_data.get('data', {})
        if not data:
            logger.error(f"❌ No data in API response")
            return None
        
        # Extract job and candidate information exactly as provided
        job_description = data.get('job_description', {})
        candidate_data = data
        
        # Clean job summary (remove HTML)
        job_summary = job_description.get('summary', '')
        clean_summary = clean_html(job_summary)
        
        # Build output strictly from API fields (no jobid/appid - only used in call.py)
        extracted_data = {
            "status": api_data.get("status"),
            "message": api_data.get("message"),
            "job_title": job_description.get("title"),
            "job_summary": clean_summary,
            "job_skills": job_description.get("skills"),
            "candidate_name": candidate_data.get("candidate_name"),
            "candidate_email": candidate_data.get("candidate_email"),
            "cv_skills": candidate_data.get("candidate_cv", {}).get("skills"),
            "cv_file": candidate_data.get("candidate_cv", {}).get("career_summary")
        }
        
        logger.info(f"✅ Successfully fetched data for {extracted_data['candidate_name']} - {extracted_data['job_title']}")
        return extracted_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching job and candidate data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Interview completion phrases (from prompt.py instructions)
INTERVIEW_COMPLETION_PHRASES = [
    "that concludes our interview",
    "concludes our interview",
    "thank you for taking the time",
    "we'll be in touch",
    "we will be in touch",
    "that concludes the interview",
    "interview is complete",
    "interview has concluded",
    "thank you for your time today",
    "this concludes our interview",
    "interview is now complete"
]

# Track if evaluation has been triggered (to prevent duplicates)
evaluation_triggered = {}

# Store current participant identity per session (for completion detection)
current_participant_identities = {}

# Global context storage for completion trigger (set in entrypoint)
completion_context = {
    "ctx": None,
    "session": None,
    "participant_recording_times": None,
    "participant_recording_tasks": None,
    "participant_recording_files": None,
    "participant_transcript_files": None,
    "background_tasks": None
}


@function_tool
async def lookup_weather(
        context: RunContext,
        location: str,
):
    """Used to look up weather information."""
    return {"weather": "sunny", "temperature": 70}


async def write_transcript(session: AgentSession) -> Optional[str]:
    """Save transcript to file and return the absolute path"""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = "."
    filename = os.path.join(save_directory, f"transcript_{current_date}.json")
    # Convert to absolute path
    absolute_path = os.path.abspath(filename)

    # Save both session history and our real-time transcript data
    full_transcript = {
        "session_history": session.history.to_dict(),
        "real_time_transcript": transcript_data,
        "timestamp": current_date,
        "interview_metadata": interview_metadata
    }

    with open(absolute_path, 'w') as f:
        json.dump(full_transcript, f, indent=2)

    logger.info(f"✅ Transcript saved: {absolute_path}")
    return absolute_path




async  def evaluate_interview(
        links_file_path,
        job_description=None,
        candidate_cv=None,
        candidate_info=None,
        evaluation_instruction=None,
        use_api=False
):
    from evaluate import evaluate_interview_comprehensive
    from prompt import load_transcript_from_file

    interview_transcript = load_transcript_from_file(links_file_path)

    # Step 4: Load job description, CV, and candidate info if not provided
    participant_identity = None
    if not job_description or not candidate_cv or not candidate_info:

        filename = os.path.basename(links_file_path)
        # Remove _links.json suffix
        if filename.endswith('_links.json'):
            name_without_suffix = filename[:-11]  # Remove '_links.json'
            # Remove timestamp (last two parts: YYYYMMDD_HHMMSS)
            parts = name_without_suffix.split('_')
            if len(parts) >= 3:
                # Take all parts except the last two (date and time)
                participant_identity = '_'.join(parts[:-2])
            else:
                participant_identity = name_without_suffix
        else:
            # Fallback: just remove .json if present
            participant_identity = filename.replace('.json', '').replace('_links', '')

        # Try participant-specific config first
        participant_config_file = f"config_{participant_identity}.json"
        config_file = None

        if os.path.exists(participant_config_file):
            config_file = participant_config_file
        elif os.path.exists("config.json"):
            config_file = "config.json"

        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if not job_description:
                    job_description = config.get("job_description", {})
                if not candidate_cv:
                    candidate_cv = config.get("candidate_cv", {})
                if not candidate_info:
                    candidate_info = config.get("candidate_info", {})

    # Use defaults if still empty
    if not candidate_info:
        participant_id = participant_identity if participant_identity else "unknown"
        candidate_info = {
            "candidate_id": participant_id,
            "candidate_email": f"{participant_id}@example.com",
            "candidate_name": participant_id.replace("_", " ").title(),
            "job_id": "unknown"
        }

    if not job_description:
        job_description = {
            "title": os.getenv("JOB_TITLE", "AI Engineer"),
            "department": os.getenv("JOB_DEPARTMENT", "Technology"),
            "location": os.getenv("JOB_LOCATION", "Remote"),
            "employment_type": os.getenv("JOB_EMPLOYMENT_TYPE", "Full-time"),
            "summary": os.getenv("JOB_SUMMARY", "We are seeking a qualified candidate for this position."),
            "responsibilities": [],
            "requirements": [],
            "skills": []
        }

    evaluation = None

    # Run the blocking synchronous evaluation in a thread pool to avoid blocking the event loop
    # This allows other async operations to continue in parallel
    evaluation = await asyncio.to_thread(
        evaluate_interview_comprehensive,
        job_description=job_description,
        candidate_cv=candidate_cv or {},
        candidate_info=candidate_info,
        interview_transcript=interview_transcript,
        evaluation_instruction=evaluation_instruction,
        use_api=use_api
    )

    return f"   Overall Score: {evaluation.get('overall_score', 'N/A')}/10", f"   Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10"


async def cleanup_local_files(combined_video_path: str, transcript_path: str,
                              video_path: Optional[str] = None,
                              audio_path: Optional[str] = None) -> bool:
    """
    Delete local files after successful upload to Azure

    Args:
        combined_video_path: Path to combined video file
        transcript_path: Path to transcript JSON file
        video_path: Optional path to original video file
        audio_path: Optional path to original audio file

    Returns:
        True if all files were deleted successfully, False otherwise
    """
    files_to_delete = [
        combined_video_path,
        transcript_path
    ]

    if video_path:
        files_to_delete.append(video_path)
    if audio_path:
        files_to_delete.append(audio_path)

    all_deleted = True
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"🗑️  Deleted: {file_path}")
            except Exception as e:
                logger.error(f"❌ Failed to delete {file_path}: {e}")
                all_deleted = False
        elif file_path:
            logger.warning(f"⚠️  File not found: {file_path}")

    return all_deleted


async def update_user_transcript(text: str):
    """Update user transcript in real-time"""
    transcript_data["user_transcript"] = text
    transcript_data["conversation_history"].append({
        "speaker": "user",
        "text": text,
        "timestamp": datetime.now().isoformat()
    })
    logger.info(f"User: {text}")


async def update_agent_transcript(text: str, participant_identity: str = None):
    """Update agent transcript in real-time and detect interview completion"""
    transcript_data["agent_transcript"] = text
    transcript_data["conversation_history"].append({
        "speaker": "agent",
        "text": text,
        "timestamp": datetime.now().isoformat()
    })
    logger.info(f"Agent: {text}")
    
    # Check if agent has signaled interview completion
    # Use provided participant_identity or try to get from global context
    if not participant_identity:
        # Try to get from global context (use first available if multiple)
        if current_participant_identities:
            participant_identity = list(current_participant_identities.values())[0] if isinstance(current_participant_identities, dict) else None
    
    if participant_identity:
        text_lower = text.lower()
        for phrase in INTERVIEW_COMPLETION_PHRASES:
            if phrase in text_lower:
                logger.info(f"🎯 Interview completion detected! Phrase: '{phrase}'")
                # Trigger evaluation workflow if not already triggered
                if participant_identity not in evaluation_triggered or not evaluation_triggered[participant_identity]:
                    logger.info(f"🚀 Auto-triggering evaluation workflow for {participant_identity}")
                    evaluation_triggered[participant_identity] = True
                    # Trigger the evaluation workflow asynchronously
                    if all([completion_context["ctx"], completion_context["session"]]):
                        asyncio.create_task(trigger_evaluation_on_completion(
                            participant_identity,
                            completion_context["ctx"],
                            completion_context["session"],
                            completion_context["participant_recording_times"],
                            completion_context["participant_recording_tasks"],
                            completion_context["participant_recording_files"],
                            completion_context["participant_transcript_files"],
                            completion_context["background_tasks"]
                        ))
                    else:
                        logger.warning(f"⚠️ Completion context not ready, cannot trigger disconnect workflow")
                break


async def trigger_evaluation_on_completion(participant_identity: str, ctx: JobContext, session: AgentSession,
                                          participant_recording_times: dict, participant_recording_tasks: dict,
                                          participant_recording_files: dict, participant_transcript_files: dict,
                                          background_tasks: list):
    """
    Trigger disconnect workflow when AI agent signals interview completion.
    This will save transcript, upload files, run evaluation, and disconnect the participant.
    """
    logger.info(f"🎯 Interview completion detected for {participant_identity}, triggering disconnect workflow...")
    
    # Mark evaluation as triggered to prevent duplicates
    evaluation_triggered[participant_identity] = True
    
    # Wait a bit for any final transcript updates
    await asyncio.sleep(2)
    
    # Get the participant from the room
    participant = None
    for remote_participant in ctx.room.remote_participants.values():
        if remote_participant.identity == participant_identity:
            participant = remote_participant
            break
    
    if not participant:
        logger.warning(f"⚠️ Participant {participant_identity} not found in room, cannot trigger disconnect workflow")
        return
    
    logger.info(f"🔴 Triggering disconnect workflow for {participant_identity} (AI detected interview completion)")
    
    # Manually trigger the disconnect workflow by calling the same logic
    disconnect_time = datetime.now()
    
    # Set SHARED end time IMMEDIATELY
    if participant_identity in participant_recording_times:
        participant_recording_times[participant_identity]['end'] = disconnect_time
        logger.info(f"🎬 SHARED END time locked: {disconnect_time}")
        
        # Calculate final duration
        start = participant_recording_times[participant_identity]['start']
        duration = (disconnect_time - start).total_seconds()
        logger.info(f"📊 Final recording duration: {duration:.1f} seconds")
        logger.info(f"🎬 Both video and audio will use this EXACT timeline")
    
    # Cancel recording tasks to stop immediately (not wait for streams)
    if participant_identity in participant_recording_tasks:
        tasks = participant_recording_tasks[participant_identity]
        for task_name, task in tasks.items():
            if task and not task.done():
                task.cancel()
                logger.info(f"🛑 Cancelled {task_name} recording task")
        logger.info(f"✅ Both recordings stopped at SAME TIME: {disconnect_time}")
    
    # Trigger the save and upload workflow (same as disconnect handler)
    async def save_and_upload_workflow():
        """Save transcript, upload to Azure, save links, then evaluate automatically"""
        try:
            logger.info(f"💾 Saving transcript for {participant_identity}...")
            transcript_path = await write_transcript(session)
            
            # Run evaluation directly (same as manual disconnect handler)
            logger.info(f"📊 Starting evaluation for {participant_identity}...")
            try:
                ev = await evaluate_interview(transcript_path)
                logger.info(f"✅ Evaluation completed for {participant_identity}: {ev}")
            except Exception as eval_error:
                logger.error(f"❌ Evaluation error for {participant_identity}: {eval_error}")
                import traceback
                traceback.print_exc()
            
            if transcript_path:
                participant_transcript_files[participant_identity] = transcript_path
                
                # Wait for video to be ready (with timeout)
                logger.info(f"⏳ Waiting for video to be ready...")
                max_wait = 15  # seconds
                waited = 0
                combined_video_path = None
                
                while waited < max_wait:
                    if participant_identity in participant_recording_files:
                        combined_video_path = participant_recording_files[participant_identity].get('combined')
                        if combined_video_path and os.path.exists(combined_video_path):
                            logger.info(f"✅ Video ready: {combined_video_path}")
                            break
                    await asyncio.sleep(1)
                    waited += 1
                
                # Upload to Azure in background thread (daemon - won't block exit)
                def upload_and_save_links_background():
                    """Upload to Azure, save links JSON, then automatically evaluate"""
                    try:
                        import time
                        time.sleep(0.5)  # Brief delay
                        
                        from azure_storage import upload_video_to_azure, upload_transcript_to_azure
                        from save_links import save_azure_links
                        
                        # Capture transcript_path from outer scope for evaluation
                        local_transcript_path = transcript_path
                        
                        # Upload transcript to Azure
                        logger.info(f"📤 Uploading transcript to Azure...")
                        transcript_result = upload_transcript_to_azure(transcript_path)
                        
                        if not transcript_result.get('success'):
                            logger.error(f"❌ Failed to upload transcript: {transcript_result.get('error')}")
                            return
                        
                        transcript_url = transcript_result['blob_url']
                        logger.info(f"✅ Transcript uploaded: {transcript_url[:80]}...")
                        
                        # Upload video if ready
                        video_url = None
                        if combined_video_path and os.path.exists(combined_video_path):
                            logger.info(f"📤 Uploading video to Azure...")
                            video_result = upload_video_to_azure(combined_video_path)
                            
                            if video_result.get('success'):
                                video_url = video_result['blob_url']
                                logger.info(f"✅ Video uploaded: {video_url[:80]}...")
                            else:
                                logger.warning(f"⚠️ Video upload failed: {video_result.get('error')}")
                                video_url = "pending"
                        else:
                            logger.warning(f"⚠️ Video not ready yet, using placeholder")
                            video_url = "pending"
                        
                        # Save links JSON
                        logger.info(f"💾 Saving Azure links to JSON file...")
                        result = save_azure_links(
                            participant_identity=participant_identity,
                            video_url=video_url,
                            transcript_url=transcript_url,
                            auto_evaluate=False
                        )
                        
                        if result and result.get('success'):
                            links_file = result.get('links_file')
                            logger.info(f"✅ Links saved: {links_file}")
                            
                            # Automatically run evaluation with the saved links file in background
                            def run_evaluation_in_thread():
                                """Run evaluation in a separate async context"""
                                try:
                                    logger.info(f"📊 Starting automatic interview evaluation...")
                                    
                                    # Use links_file if available, otherwise fall back to transcript_path
                                    evaluation_file = links_file if links_file and os.path.exists(links_file) else local_transcript_path
                                    logger.info(f"📊 Using file for evaluation: {evaluation_file}")
                                    
                                    # Create new event loop for this thread and run evaluation
                                    import asyncio
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    
                                    try:
                                        # Run the async evaluation function
                                        evaluation_result = loop.run_until_complete(evaluate_interview(evaluation_file))
                                        
                                        logger.info(f"✅ Interview evaluation completed successfully!")
                                        if isinstance(evaluation_result, tuple):
                                            logger.info(f"📊 {evaluation_result[0]}")
                                            logger.info(f"📊 {evaluation_result[1]}")
                                        else:
                                            logger.info(f"📊 Evaluation Result: {evaluation_result}")
                                    finally:
                                        loop.close()
                                except Exception as e:
                                    logger.error(f"❌ Evaluation error: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # Start evaluation in a separate daemon thread (fire and forget)
                            eval_thread = threading.Thread(target=run_evaluation_in_thread, daemon=True, name="EvaluationThread")
                            eval_thread.start()
                            logger.info(f"🚀 Evaluation started automatically in background thread")
                            
                            # Also try to trigger evaluation via API as fallback
                            try:
                                import requests
                                api_url = os.getenv('API_SERVER_URL', 'http://localhost:5001')
                                logger.info(f"🔔 Also triggering evaluation via API as fallback...")
                                
                                requests.post(
                                    f"{api_url}/api/evaluations/evaluate_from_links",
                                    json={
                                        "links_file": links_file,
                                        "participant_identity": participant_identity
                                    },
                                    timeout=2
                                )
                                logger.info(f"✅ Evaluation API also triggered")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to trigger evaluation API (non-critical): {e}")
                        else:
                            logger.error(f"❌ Failed to save links")
                    
                    except Exception as e:
                        logger.error(f"❌ Upload error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Start upload in daemon thread
                upload_thread = threading.Thread(target=upload_and_save_links_background, daemon=True, name="UploadThread")
                upload_thread.start()
                logger.info(f"🚀 Upload started in background (independent daemon thread)")
            else:
                logger.error(f"❌ Failed to save transcript")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Start the workflow and wait for it to complete (don't close session yet)
    logger.info(f"🚀 Starting disconnect workflow for {participant_identity}...")
    await save_and_upload_workflow()
    logger.info(f"✅ Disconnect workflow completed for {participant_identity}")
    
    # Don't close the session here - let it close naturally when participant disconnects
    # The workflow has completed, evaluation has run, files are uploaded
    logger.info(f"ℹ️ Workflow complete - participant can disconnect naturally")


def create_transcription_node(participant_identity: str = None):
    """Create a transcription node with participant identity context"""
    async def transcription_node(text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]:
        """Process transcription with real-time updates"""
        full_text = ""
        async for delta in text:
            cleaned_delta = delta.replace("😘", "")
            full_text += cleaned_delta
            # Update agent transcript in real-time with participant identity for completion detection
            # Use closure participant_identity or try to get from global
            identity_to_use = participant_identity
            if not identity_to_use and current_participant_identities:
                # Try to get from global context (use first available)
                identity_to_use = list(current_participant_identities.values())[0] if isinstance(current_participant_identities, dict) else None
            await update_agent_transcript(full_text, participant_identity=identity_to_use)
            yield cleaned_delta
    return transcription_node


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""

    async def shutdown_callback():
        """Cleanup on shutdown - exit immediately, daemon threads handle rest"""
        logger.info("🛑 Agent session ending - uploads continue in background")
        logger.info("✅ Agent session ended - process will exit now")

    ctx.add_shutdown_callback(shutdown_callback)
    await ctx.connect()

    # Extract jobid and appid from room name (only used to pass to call.py via fetch_job_and_candidate_data)
    # Room name format: voice_assistant_room_{jobid}_{appid}_{suffix} or voice_assistant_room_{suffix}
    jobid = None
    appid = None
    
    room_name = ctx.room.name if hasattr(ctx.room, 'name') else None
    if room_name:
        # Try to extract jobid and appid from room name
        # Format: voice_assistant_room_{jobid}_{appid}_{suffix}
        parts = room_name.split('_')
        if len(parts) >= 5 and parts[0] == 'voice' and parts[1] == 'assistant' and parts[2] == 'room':
            try:
                # Check if parts[3] and parts[4] are numeric (jobid and appid)
                if parts[3].isdigit() and parts[4].isdigit():
                    jobid = parts[3]
                    appid = parts[4]
                    logger.info(f"📋 Extracted from room name - jobid: {jobid}, appid: {appid}")
            except (ValueError, IndexError):
                pass
    
    # Fetch real job and candidate data (jobid/appid only used here to call call.py)
    interview_data = fetch_job_and_candidate_data(jobid, appid)
    
    # Store interview metadata globally for transcript (no jobid/appid stored)
    if interview_data:
        interview_metadata["job_title"] = interview_data.get("job_title")
        interview_metadata["candidate_name"] = interview_data.get("candidate_name")
    
    # Prepare instruction parameters - only use API data, no defaults
    if not interview_data:
        logger.error("❌ No interview data from API; cannot proceed without required data")
        raise ValueError("Interview data is required from API.")
    
    # Extract only the required fields from API data
    job_title = interview_data.get("job_title") or ""
    job_summary = interview_data.get("job_summary") or ""
    job_skills = interview_data.get("job_skills") or []
    candidate_name = interview_data.get("candidate_name") or ""
    cv_skills = interview_data.get("cv_skills") or []
    
    logger.info(f"✅ Using API interview data:")
    logger.info(f"   Job: {job_title}")
    logger.info(f"   Candidate: {candidate_name}")
    logger.info(f"   Job Skills: {job_skills}")
    logger.info(f"   CV Skills: {cv_skills}")

    # Create agent with instructions using only API data
    agent = Agent(
        instructions=get_instruction(
            job_title=job_title,
            job_summary=job_summary,
            job_skills=job_skills,
            candidate_name=candidate_name,
            cv_skills=cv_skills
        ),
        tools=[lookup_weather],
    )

    # Create session with real-time transcription support and completion detection
    # We'll update the transcription node after we get participant identity
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        use_tts_aligned_transcript=True,
    )

    # elevenlabs/eleven_turbo_v2_5:cjVigY5qzO86Huf0OWal

    # Shared timestamps and tasks for perfect video/audio sync
    participant_recording_times = {}
    participant_recording_tasks = {}  # Track recording tasks for each participant
    participant_recording_files = {}  # Track video/audio filenames for each participant
    participant_transcript_files = {}  # Track transcript files for each participant
    participant_upload_in_progress = {}  # Track if upload is in progress to prevent duplicates
    background_tasks = []  # Track all background tasks for cleanup on shutdown
    
    # Store context for completion trigger (after all variables are defined)
    completion_context["ctx"] = ctx
    completion_context["session"] = session
    completion_context["participant_recording_times"] = participant_recording_times
    completion_context["participant_recording_tasks"] = participant_recording_tasks
    completion_context["participant_recording_files"] = participant_recording_files
    completion_context["participant_transcript_files"] = participant_transcript_files
    completion_context["background_tasks"] = background_tasks

    async def upload_and_cleanup_workflow(participant_identity: str, combined_video_path: str,
                                          transcript_path: str):
        """
        Complete workflow: Upload video and transcript to Azure, save links, then cleanup local files

        Args:
            participant_identity: Identity of the participant
            combined_video_path: Path to the combined video file
            transcript_path: Path to the transcript JSON file
        """
        # Prevent duplicate uploads
        if participant_identity in participant_upload_in_progress:
            if participant_upload_in_progress[participant_identity]:
                logger.info(f"⚠️ Upload already in progress for {participant_identity}, skipping...")
                return

        participant_upload_in_progress[participant_identity] = True

        try:
            logger.info(f"🚀 Starting upload and cleanup workflow for {participant_identity}")

            # Step 1: Upload combined video to Azure
            logger.info(f"📤 Uploading combined video to Azure...")
            video_upload_result = upload_video_to_azure(combined_video_path)

            if not video_upload_result.get('success'):
                logger.error(f"❌ Failed to upload video: {video_upload_result.get('error')}")
                return

            video_url = video_upload_result['blob_url']
            logger.info(f"✅ Video uploaded successfully: {video_url}")

            # Step 2: Upload transcript to Azure
            logger.info(f"📤 Uploading transcript to Azure...")
            transcript_upload_result = upload_transcript_to_azure(transcript_path)

            if not transcript_upload_result.get('success'):
                logger.error(f"❌ Failed to upload transcript: {transcript_upload_result.get('error')}")
                return

            transcript_url = transcript_upload_result['blob_url']
            logger.info(f"✅ Transcript uploaded successfully: {transcript_url}")

            # Step 3: Save Azure links to file
            logger.info(f"💾 Saving Azure links...")
            links_filename = None
            try:
                links_result = save_azure_links(participant_identity, video_url, transcript_url, auto_evaluate=False)
                if links_result and isinstance(links_result, dict) and links_result.get('links_file'):
                    links_filename = links_result.get('links_file')
                    logger.info(f"✅ Links saved successfully: {links_filename}")
                else:
                    logger.warning(f"⚠️ Links result format unexpected: {links_result}")
                    # Fallback: construct filename manually
                    links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
                    logger.info(f"✅ Using fallback links filename: {links_filename}")
            except Exception as e:
                logger.error(f"❌ Error saving links: {e}")
                import traceback
                traceback.print_exc()
                # Still try to construct filename for evaluation
                links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
                logger.info(f"⚠️ Using fallback links filename after error: {links_filename}")

            # Step 3.5: Evaluation happens in disconnect handler, skip duplicate evaluation here
            logger.info(f"ℹ️ Evaluation will run automatically in disconnect handler")

            # Step 4: Get original video and audio paths for cleanup
            original_video_path = None
            original_audio_path = None
            if participant_identity in participant_recording_files:
                original_video_path = participant_recording_files[participant_identity].get('video')
                original_audio_path = participant_recording_files[participant_identity].get('audio')

            # Step 5: Cleanup local files
            logger.info(f"🧹 Cleaning up local files...")
            cleanup_success = await cleanup_local_files(
                combined_video_path=combined_video_path,
                transcript_path=transcript_path,
                video_path=original_video_path,
                audio_path=original_audio_path
            )

            if cleanup_success:
                logger.info(f"✅ All local files cleaned up successfully!")
            else:
                logger.warning(f"⚠️  Some files could not be deleted")

            logger.info(f"🎉 Upload and cleanup workflow completed for {participant_identity}")

        except Exception as e:
            logger.error(f"❌ Error in upload and cleanup workflow: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset upload flag
            participant_upload_in_progress[participant_identity] = False

    async def combine_video_audio(participant_identity: str):
        """Combine video and audio files into a single MP4 with perfect sync"""
        if participant_identity not in participant_recording_files:
            logger.warning(f"⚠️ No recording files found for {participant_identity}")
            return

        files = participant_recording_files[participant_identity]
        video_file = files.get('video')
        audio_file = files.get('audio')

        if not video_file or not audio_file:
            logger.warning(f"⚠️ Missing video or audio file for {participant_identity}")
            logger.info(f"   Video: {video_file}")
            logger.info(f"   Audio: {audio_file}")
            return

        # Check if both files exist
        if not os.path.exists(video_file) or not os.path.exists(audio_file):
            logger.error(f"❌ Files not found:")
            logger.error(f"   Video: {os.path.exists(video_file)} - {video_file}")
            logger.error(f"   Audio: {os.path.exists(audio_file)} - {audio_file}")
            return

        # Create combined filename
        os.makedirs("recordings/combined", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = f"recordings/combined/{participant_identity}_{timestamp}_COMBINED.mp4"

        logger.info(f"🎬 Combining video and audio...")
        logger.info(f"   Video: {video_file}")
        logger.info(f"   Audio: {audio_file}")
        logger.info(f"   Output: {combined_file}")

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
                logger.info(f"✅ Successfully combined video and audio!")
                logger.info(f"📁 Combined file: {combined_file}")

                # Check file size
                size_mb = os.path.getsize(combined_file) / (1024 * 1024)
                logger.info(f"📊 File size: {size_mb:.2f} MB")

                # Track combined file for this participant
                if participant_identity not in participant_recording_files:
                    participant_recording_files[participant_identity] = {}
                participant_recording_files[participant_identity]['combined'] = combined_file
                logger.info(f"✅ Video tracked and ready for upload (will be uploaded by disconnect handler)")
            else:
                logger.error(f"❌ ffmpeg error:")
                logger.error(f"   {result.stderr}")

        except FileNotFoundError:
            logger.error(f"❌ ffmpeg not found! Install with: brew install ffmpeg")
        except Exception as e:
            logger.error(f"❌ Error combining files: {e}")
            import traceback
            traceback.print_exc()

    # Record video with TIMESTAMP TRACKING for perfect audio sync
    async def handle_video_recording(video_track: rtc.Track, participant_identity: str):
        """Record video with timestamp tracking - fills gaps for perfect sync with audio"""
        video_stream = rtc.VideoStream(video_track)
        frames_with_timestamps = []  # Store (timestamp, frame) tuples
        start_time = None
        end_time = None
        width = None
        height = None
        frame_count = 0
        valid_frame_count = 0

        try:
            logger.info(f"🎥 Starting VIDEO recording for {participant_identity}")
            logger.info(f"🎥 Mode: TIMESTAMP TRACKING (for perfect audio sync)")

            # STEP 1: Collect ALL frames with timestamps
            async for event in video_stream:
                try:
                    video_frame: rtc.VideoFrame = event.frame
                    frame_count += 1
                    current_time = datetime.now()

                    # Use SHARED start time for this participant
                    if participant_identity not in participant_recording_times:
                        participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
                        logger.info(f"🎬 Recording START time set: {current_time}")

                    if start_time is None:
                        start_time = participant_recording_times[participant_identity]['start']

                    # Convert frame for recording (RGBA → BGR)
                    frame_rgba = video_frame.convert(rtc.VideoBufferType.RGBA)
                    frame_width = frame_rgba.width
                    frame_height = frame_rgba.height

                    # Skip frames with invalid dimensions (corrupted frames)
                    if frame_width < 100 or frame_height < 100:
                        logger.debug(f"⚠️ Skipping corrupted frame {frame_count}: {frame_width}x{frame_height}")
                        continue

                    # Capture width/height from FIRST valid frame only
                    if width is None:
                        width = frame_width
                        height = frame_height
                        logger.info(f"🎥 Detected resolution: {width}x{height}")

                    # Resize frames to match target resolution instead of skipping
                    buffer = frame_rgba.data
                    img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape((frame_height, frame_width, 4))

                    # Resize if resolution changed
                    if frame_width != width or frame_height != height:
                        logger.debug(
                            f"🔄 Resizing frame {frame_count} from {frame_width}x{frame_height} to {width}x{height}")
                        img_rgba = cv2.resize(img_rgba, (width, height), interpolation=cv2.INTER_LINEAR)

                    img_array = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

                    # Store frame with its timestamp
                    frames_with_timestamps.append((current_time, img_array.copy()))
                    valid_frame_count += 1

                    # Log progress every 30 frames
                    if valid_frame_count % 30 == 0:
                        elapsed = (current_time - start_time).total_seconds()
                        logger.info(f"🎥 Recorded {valid_frame_count} valid frames ({elapsed:.1f}s)")

                except Exception as frame_error:
                    logger.error(f"❌ Error processing frame {frame_count}: {frame_error}")
                    continue

        except asyncio.CancelledError:
            logger.info(f"🛑 Video recording cancelled (participant disconnected)")
            # Use SHARED end time set by disconnect handler
        except Exception as e:
            logger.error(f"❌ Video stream error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await video_stream.aclose()

            # Use SHARED end time if set, otherwise use now
            if participant_identity in participant_recording_times:
                end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
            else:
                end_time = datetime.now()

            # Update SHARED end time
            if participant_identity in participant_recording_times:
                participant_recording_times[participant_identity]['end'] = end_time
                logger.info(f"🎬 Video recording END time set: {end_time}")

            # STEP 2: Create video with EXACT duration matching audio (fill gaps with duplicate frames)
            if start_time and end_time and len(frames_with_timestamps) > 0 and width and height:
                # Use SHARED timestamps if available (both tracks might still be recording)
                if participant_identity in participant_recording_times:
                    shared_start = participant_recording_times[participant_identity]['start']
                    shared_end = participant_recording_times[participant_identity]['end'] or end_time
                    start_time = shared_start
                    end_time = shared_end

                total_duration = (end_time - start_time).total_seconds()
                target_fps = 30.0  # Target FPS for smooth video

                logger.info(f"✅ Frame collection complete!")
                logger.info(f"✅ Valid frames collected: {len(frames_with_timestamps)}")
                logger.info(f"✅ Total recording duration: {total_duration:.1f} seconds")
                logger.info(f"📐 Video resolution: {width}x{height}")

                # Validate resolution
                if width < 100 or height < 100:
                    logger.error(f"❌ Invalid resolution: {width}x{height}")
                    return

                # STEP 3: Generate frames at regular intervals (30 FPS) to match audio duration
                target_frame_count = int(total_duration * target_fps)
                logger.info(f"🎯 Target frames for {total_duration:.1f}s @ {target_fps} FPS: {target_frame_count}")

                output_frames = []
                frame_idx = 0

                for i in range(target_frame_count):
                    # Calculate target time for this frame
                    target_time = start_time + (i / target_fps) * timedelta(seconds=1)

                    # Find closest actual frame to this target time
                    while frame_idx < len(frames_with_timestamps) - 1:
                        current_frame_time = frames_with_timestamps[frame_idx][0]
                        next_frame_time = frames_with_timestamps[frame_idx + 1][0]

                        # If next frame is closer to target, move forward
                        if abs((next_frame_time - target_time).total_seconds()) < abs(
                                (current_frame_time - target_time).total_seconds()):
                            frame_idx += 1
                        else:
                            break

                    # Use the closest frame
                    output_frames.append(frames_with_timestamps[frame_idx][1])

                logger.info(f"✅ Generated {len(output_frames)} frames (filled gaps for continuous playback)")

                # STEP 4: Write ALL frames to video
                os.makedirs("recordings/video", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"recordings/video/{participant_identity}_{timestamp}_VIDEO.mp4"

                # Use H.264 codec (avc1) for browser compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(video_filename, fourcc, target_fps, (width, height))

                if not video_writer.isOpened():
                    logger.error(f"❌ Failed to open video writer!")
                    return

                logger.info(f"🎥 Writing {len(output_frames)} frames to: {video_filename}")
                logger.info(f"🎥 Settings: {width}x{height} @ {target_fps} FPS")

                # Write ALL frames
                for i, frame in enumerate(output_frames):
                    video_writer.write(frame)
                    if (i + 1) % 100 == 0 or (i + 1) == len(output_frames):
                        logger.info(f"🎥 Written {i + 1}/{len(output_frames)} frames...")

                video_writer.release()
                logger.info(f"✅ Video writer released")

                logger.info(f"🎉 Video saved successfully!")
                logger.info(f"📁 File: {video_filename}")
                logger.info(f"📊 Actual frames captured: {len(frames_with_timestamps)}")
                logger.info(f"📊 Output frames (with gap filling): {len(output_frames)}")
                logger.info(f"⏱️  Video duration: {total_duration:.1f} seconds")
                logger.info(f"🎬 FPS: {target_fps}")
                logger.info(f"🔄 Gap filling: {len(output_frames) - len(frames_with_timestamps)} duplicate frames")
                logger.info(f"📺 Video duration will MATCH audio duration perfectly!")

                # Track video filename for combining later
                if participant_identity not in participant_recording_files:
                    participant_recording_files[participant_identity] = {}
                participant_recording_files[participant_identity]['video'] = video_filename
                logger.info(f"✅ Video filename tracked for combining")

                # Check if audio is also done - if yes, combine them!
                if 'audio' in participant_recording_files.get(participant_identity, {}):
                    logger.info(f"🎬 Both video and audio ready - combining now!")
                    task = asyncio.create_task(combine_video_audio(participant_identity))
                    background_tasks.append(task)
            else:
                logger.warning(f"⚠️ No valid frames were recorded!")

    # Record audio with SHARED TIMESTAMPS for perfect video sync
    async def handle_audio_recording(audio_track: rtc.Track, participant_identity: str):
        """Record audio with shared timestamps - matches video duration"""
        audio_stream = rtc.AudioStream(audio_track)
        audio_frames = []
        audio_filename = None
        sample_rate = None
        num_channels = None
        audio_frame_count = 0
        start_time = None

        try:
            logger.info(f"🎤 Starting AUDIO recording for {participant_identity}")

            async for event in audio_stream:
                current_time = datetime.now()
                audio_frame: rtc.AudioFrame = event.frame
                audio_frame_count += 1

                # Use SHARED start time for this participant
                if participant_identity not in participant_recording_times:
                    participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
                    logger.info(f"🎬 Recording START time set: {current_time}")

                if start_time is None:
                    start_time = participant_recording_times[participant_identity]['start']

                # Get audio properties from first frame
                if sample_rate is None:
                    sample_rate = audio_frame.sample_rate
                    num_channels = audio_frame.num_channels

                    os.makedirs("recordings/audio", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"recordings/audio/{participant_identity}_{timestamp}_AUDIO.wav"

                    logger.info(f"🎤 Recording to: {audio_filename}")
                    logger.info(f"🎤 Sample rate: {sample_rate}Hz, Channels: {num_channels}")

                # Store audio data (no processing, keep quality)
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
                audio_frames.append(audio_data.tobytes())

                # Log progress
                if audio_frame_count % 100 == 0:
                    logger.info(f"🎤 Audio: {audio_frame_count} frames")

        except asyncio.CancelledError:
            logger.info(f"🛑 Audio recording cancelled (participant disconnected)")
            # Use SHARED end time set by disconnect handler
        except Exception as e:
            logger.error(f"❌ Audio error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await audio_stream.aclose()

            # Use SHARED end time if set, otherwise use now
            if participant_identity in participant_recording_times:
                end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
            else:
                end_time = datetime.now()

            # Update SHARED end time
            if participant_identity in participant_recording_times:
                # Use the LATEST end time between video and audio
                current_end = participant_recording_times[participant_identity]['end']
                if current_end is None or end_time > current_end:
                    participant_recording_times[participant_identity]['end'] = end_time
                logger.info(f"🎬 Audio recording END time set: {end_time}")

            # Save audio to WAV (lossless)
            if audio_frames and sample_rate:
                try:
                    with wave.open(audio_filename, 'wb') as wav_file:
                        wav_file.setnchannels(num_channels)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(b''.join(audio_frames))

                    # Log with shared timestamps
                    if participant_identity in participant_recording_times:
                        shared_start = participant_recording_times[participant_identity]['start']
                        shared_end = participant_recording_times[participant_identity]['end']
                        if shared_start and shared_end:
                            duration = (shared_end - shared_start).total_seconds()
                            logger.info(f"✅ Audio saved: {audio_filename} ({audio_frame_count} frames)")
                            logger.info(f"🎬 Audio uses SHARED timeline: {duration:.1f} seconds")

                    # Track audio filename for combining later
                    if participant_identity not in participant_recording_files:
                        participant_recording_files[participant_identity] = {}
                    participant_recording_files[participant_identity]['audio'] = audio_filename
                    logger.info(f"✅ Audio filename tracked for combining")

                    # Check if video is also done - if yes, combine them!
                    if 'video' in participant_recording_files.get(participant_identity, {}):
                        logger.info(f"🎬 Both video and audio ready - combining now!")
                        task = asyncio.create_task(combine_video_audio(participant_identity))
                        background_tasks.append(task)

                except Exception as e:
                    logger.error(f"❌ Error saving audio: {e}")
                    import traceback
                    traceback.print_exc()

    # Participant disconnect handler - ensures both video and audio stop together
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """Handle participant disconnect - IMMEDIATELY stop both video and audio"""
        disconnect_time = datetime.now()
        logger.info(f"🔴 Participant disconnected: {participant.identity}")




        # Set SHARED end time IMMEDIATELY
        if participant.identity in participant_recording_times:
            participant_recording_times[participant.identity]['end'] = disconnect_time
            logger.info(f"🎬 SHARED END time locked: {disconnect_time}")

            # Calculate final duration
            start = participant_recording_times[participant.identity]['start']
            duration = (disconnect_time - start).total_seconds()
            logger.info(f"📊 Final recording duration: {duration:.1f} seconds")
            logger.info(f"🎬 Both video and audio will use this EXACT timeline")

        # Cancel recording tasks to stop immediately (not wait for streams)
        if participant.identity in participant_recording_tasks:
            tasks = participant_recording_tasks[participant.identity]
            for task_name, task in tasks.items():
                if task and not task.done():
                    task.cancel()
                    logger.info(f"🛑 Cancelled {task_name} recording task")
            logger.info(f"✅ Both recordings stopped at SAME TIME: {disconnect_time}")


        # logger.info(f"💾 Saving transcript for {participant.identity}...")

        # Save transcript, upload to Azure, and save links - then evaluation runs automatically
        async def save_and_upload_workflow():
            """Save transcript, upload to Azure, save links, then evaluate automatically"""

            try:
                logger.info(f"💾 Saving transcript for {participant.identity}...")
                transcript_path = await write_transcript(session)

                # Run evaluation directly (same as AI-triggered completion)
                if transcript_path:
                    logger.info(f"📊 Starting evaluation for {participant.identity}...")
                    try:
                        ev = await evaluate_interview(transcript_path)
                        logger.info(f"✅ Evaluation completed for {participant.identity}: {ev}")
                    except Exception as eval_error:
                        logger.error(f"❌ Evaluation error for {participant.identity}: {eval_error}")
                        import traceback
                        traceback.print_exc()






                if transcript_path:
                    participant_transcript_files[participant.identity] = transcript_path

                    # Wait for video to be ready (with timeout)
                    logger.info(f"⏳ Waiting for video to be ready...")
                    max_wait = 15  # seconds
                    waited = 0
                    combined_video_path = None

                    while waited < max_wait:
                        if participant.identity in participant_recording_files:
                            combined_video_path = participant_recording_files[participant.identity].get('combined')
                            if combined_video_path and os.path.exists(combined_video_path):
                                logger.info(f"✅ Video ready: {combined_video_path}")
                                break
                        await asyncio.sleep(1)
                        waited += 1

                    # Upload to Azure in background thread (daemon - won't block exit)
                    def upload_and_save_links_background():
                        """Upload to Azure, save links JSON, then automatically evaluate"""
                        try:
                            import time
                            time.sleep(0.5)  # Brief delay

                            from azure_storage import upload_video_to_azure, upload_transcript_to_azure
                            from save_links import save_azure_links
                            
                            # Capture transcript_path from outer scope for evaluation
                            local_transcript_path = transcript_path

                            # Upload transcript to Azure
                            logger.info(f"📤 Uploading transcript to Azure...")
                            transcript_result = upload_transcript_to_azure(transcript_path)

                            if not transcript_result.get('success'):
                                logger.error(f"❌ Failed to upload transcript: {transcript_result.get('error')}")
                                return

                            transcript_url = transcript_result['blob_url']
                            logger.info(f"✅ Transcript uploaded: {transcript_url[:80]}...")

                            # Upload video if ready
                            video_url = None
                            if combined_video_path and os.path.exists(combined_video_path):
                                logger.info(f"📤 Uploading video to Azure...")
                                video_result = upload_video_to_azure(combined_video_path)

                                if video_result.get('success'):
                                    video_url = video_result['blob_url']
                                    logger.info(f"✅ Video uploaded: {video_url[:80]}...")
                                else:
                                    logger.warning(f"⚠️ Video upload failed: {video_result.get('error')}")
                                    video_url = "pending"
                            else:
                                logger.warning(f"⚠️ Video not ready yet, using placeholder")
                                video_url = "pending"

                            # Save links JSON (fast, no evaluation here)
                            logger.info(f"💾 Saving Azure links to JSON file...")
                            result = save_azure_links(
                                participant_identity=participant.identity,
                                video_url=video_url,
                                transcript_url=transcript_url,
                                auto_evaluate=False  # DON'T evaluate in agent - too slow
                            )

                            if result and result.get('success'):
                                links_file = result.get('links_file')
                                logger.info(f"✅ Links saved: {links_file}")

                                # Automatically run evaluation with the saved links file in background
                                def run_evaluation_in_thread():
                                    """Run evaluation in a separate async context"""
                                    try:
                                        logger.info(f"📊 Starting automatic interview evaluation...")
                                        
                                        # Use links_file if available, otherwise fall back to transcript_path
                                        evaluation_file = links_file if links_file and os.path.exists(links_file) else local_transcript_path
                                        logger.info(f"📊 Using file for evaluation: {evaluation_file}")
                                        
                                        # Create new event loop for this thread and run evaluation
                                        import asyncio
                                        # Create a new event loop for this thread
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        
                                        try:
                                            # Run the async evaluation function
                                            evaluation_result = loop.run_until_complete(evaluate_interview(evaluation_file))
                                            
                                            logger.info(f"✅ Interview evaluation completed successfully!")
                                            if isinstance(evaluation_result, tuple):
                                                logger.info(f"📊 {evaluation_result[0]}")
                                                logger.info(f"📊 {evaluation_result[1]}")
                                            else:
                                                logger.info(f"📊 Evaluation Result: {evaluation_result}")
                                        finally:
                                            # Clean up the event loop
                                            loop.close()
                                    except Exception as e:
                                        logger.error(f"❌ Evaluation error: {e}")
                                        import traceback
                                        traceback.print_exc()

                                # Start evaluation in a separate daemon thread (fire and forget)
                                eval_thread = threading.Thread(target=run_evaluation_in_thread, daemon=True, name="EvaluationThread")
                                eval_thread.start()
                                logger.info(f"🚀 Evaluation started automatically in background thread")

                                # Also try to trigger evaluation via API as fallback (fire and forget)
                                try:
                                    import requests
                                    api_url = os.getenv('API_SERVER_URL', 'http://localhost:5001')
                                    logger.info(f"🔔 Also triggering evaluation via API as fallback...")

                                    requests.post(
                                        f"{api_url}/api/evaluations/evaluate_from_links",
                                        json={
                                            "links_file": links_file,
                                            "participant_identity": participant.identity
                                        },
                                        timeout=2
                                    )
                                    logger.info(f"✅ Evaluation API also triggered")
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to trigger evaluation API (non-critical): {e}")
                            else:
                                logger.error(f"❌ Failed to save links")

                        except Exception as e:
                            logger.error(f"❌ Upload error: {e}")
                            import traceback
                            traceback.print_exc()

                    # Start upload in daemon thread (won't block exit, NOT tracked in background_tasks)
                    upload_thread = threading.Thread(target=upload_and_save_links_background, daemon=True,
                                                     name="UploadThread")
                    upload_thread.start()
                    logger.info(f"🚀 Upload started in background (independent daemon thread)")

                else:
                    logger.error(f"❌ Failed to save transcript")
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

        # Start transcript save, upload, and evaluation workflow
        task = asyncio.create_task(save_and_upload_workflow())
        background_tasks.append(task)

    # Track participant identity for completion detection
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """Track participant identity when they connect"""
        # Store in global dictionary for access from transcription node
        room_name = ctx.room.name if hasattr(ctx.room, 'name') else 'default'
        current_participant_identities[room_name] = participant.identity
        logger.info(f"👤 Participant connected: {participant.identity}")
        # Initialize evaluation tracking
        if participant.identity not in evaluation_triggered:
            evaluation_triggered[participant.identity] = False

    # Set up track subscription handler - NO SYNC, separate recordings
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
    ):
        """Handle track subscriptions - record video and audio SEPARATELY"""
        logger.info(f"Track subscribed from {participant.identity}: {track.kind}")

        # Initialize task tracking for this participant
        if participant.identity not in participant_recording_tasks:
            participant_recording_tasks[participant.identity] = {}

        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # Start video recording immediately - NO WAIT for audio
            logger.info(f"📹 Video track - starting independent recording")
            video_task = asyncio.create_task(handle_video_recording(track, participant.identity))
            participant_recording_tasks[participant.identity]['video'] = video_task

        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            # Start audio recording immediately - NO WAIT for video
            logger.info(f"🎤 Audio track - starting independent recording")
            audio_task = asyncio.create_task(handle_audio_recording(track, participant.identity))
            participant_recording_tasks[participant.identity]['audio'] = audio_task

    logger.info("Starting agent session...")
    
    # Get participant identity from room (if available)
    def get_participant_identity():
        """Get the first remote participant's identity"""
        for participant in ctx.room.remote_participants.values():
            return participant.identity
        # Try to get from global storage
        room_name = ctx.room.name if hasattr(ctx.room, 'name') else 'default'
        return current_participant_identities.get(room_name)
    
    # Create transcription node with participant identity
    participant_id = get_participant_identity()
    transcription_node_with_identity = create_transcription_node(participant_identity=participant_id)
    
    await session.start(agent=agent, room=ctx.room)
    
    # Update participant identity after session starts (in case it wasn't available before)
    async def monitor_participant_and_update():
        """Monitor for participant and update transcription node context"""
        room_name = ctx.room.name if hasattr(ctx.room, 'name') else 'default'
        while True:
            await asyncio.sleep(1)
            participant_id = get_participant_identity()
            if participant_id and current_participant_identities.get(room_name) != participant_id:
                current_participant_identities[room_name] = participant_id
                logger.info(f"📝 Updated participant identity for completion detection: {participant_id}")

    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_participant_and_update())
    background_tasks.append(monitor_task)
    
    # Monitor session history for interview completion phrases
    async def monitor_session_for_completion():
        """Monitor session history for interview completion phrases"""
        last_checked_length = 0
        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                # Get participant identity
                participant_id = get_participant_identity()
                if not participant_id:
                    continue
                
                # Check if already triggered
                if evaluation_triggered.get(participant_id, False):
                    continue
                
                # Get latest messages from session history
                try:
                    history_dict = session.history.to_dict()
                    items = history_dict.get('items', [])
                    
                    # Only check new items
                    if len(items) > last_checked_length:
                        # Check the latest assistant messages for completion phrases
                        for item in items[last_checked_length:]:
                            if item.get('type') == 'message' and item.get('role') == 'assistant':
                                content = item.get('content', [])
                                if isinstance(content, list) and len(content) > 0:
                                    text = content[0] if isinstance(content[0], str) else str(content[0])
                                elif isinstance(content, str):
                                    text = content
                                else:
                                    text = str(content)
                                
                                text_lower = text.lower()
                                for phrase in INTERVIEW_COMPLETION_PHRASES:
                                    if phrase in text_lower:
                                        logger.info(f"🎯 Interview completion detected in session history! Phrase: '{phrase}'")
                                        if not evaluation_triggered.get(participant_id, False):
                                            logger.info(f"🚀 Auto-triggering evaluation workflow for {participant_id}")
                                            evaluation_triggered[participant_id] = True
                                            if all([completion_context["ctx"], completion_context["session"]]):
                                                asyncio.create_task(trigger_evaluation_on_completion(
                                                    participant_id,
                                                    completion_context["ctx"],
                                                    completion_context["session"],
                                                    completion_context["participant_recording_times"],
                                                    completion_context["participant_recording_tasks"],
                                                    completion_context["participant_recording_files"],
                                                    completion_context["participant_transcript_files"],
                                                    completion_context["background_tasks"]
                                                ))
                                            else:
                                                logger.warning(f"⚠️ Completion context not ready, cannot trigger disconnect workflow")
                                        break
                        
                        last_checked_length = len(items)
                except Exception as e:
                    logger.debug(f"Error checking session history: {e}")
                    continue
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in completion monitor: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    # Start completion monitoring
    completion_monitor_task = asyncio.create_task(monitor_session_for_completion())
    background_tasks.append(completion_monitor_task)

    # Generate initial greeting
    await session.generate_reply(instructions="greet the user and ask about their day")
    logger.info("Agent session started successfully")


if __name__ == "__main__":
    # import sys

    # import sys
    #
    # env = sys.argv[1] if len(sys.argv) > 1 else "dev"
    # print(f"Running as: python agent1.py {env}")

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
