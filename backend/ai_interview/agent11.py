# from prompt import get_instruction
# from datetime import datetime, timedelta
# import json
# import asyncio
# import aiohttp
# import subprocess
# from livekit import api
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     JobContext,
#     RunContext,
#     WorkerOptions,
#     cli,
#     function_tool,
#     utils,
# )
# from livekit.plugins import deepgram, elevenlabs, openai, silero
# from livekit.agents.voice.room_io import RoomOutputOptions
# from dotenv import load_dotenv
# import os
# from livekit.agents import AgentSession, Agent, RoomInputOptions, inference
# from livekit.agents import ModelSettings
# from livekit import rtc
# from typing import AsyncIterable, Optional
# import logging
# from livekit import api
# from livekit.api.egress_service import EgressService, ParticipantEgressRequest
# from google.protobuf import message_factory
# import cv2
# import numpy as np
# import wave
# from azure_storage import AzureVideoStorage, upload_video_to_azure, upload_transcript_to_azure
# from save_links import save_azure_links, download_and_evaluate_from_links, evaluate_local_transcript
# import threading
# #
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv(".env")

# # Global transcript storage for real-time updates
# transcript_data = {
#     "user_transcript": "",
#     "agent_transcript": "",
#     "conversation_history": []
# }

# @function_tool
# async def lookup_weather(
#     context: RunContext,
#     location: str,
# ):
#     """Used to look up weather information."""
#     return {"weather": "sunny", "temperature": 70}


# async def write_transcript(session: AgentSession) -> Optional[str]:
#     """Save transcript to file and return the absolute path"""
#     current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_directory = "."
#     filename = os.path.join(save_directory, f"transcript_{current_date}.json")
#     # Convert to absolute path
#     absolute_path = os.path.abspath(filename)

#     # Save both session history and our real-time transcript data
#     full_transcript = {
#         "session_history": session.history.to_dict(),
#         "real_time_transcript": transcript_data,
#         "timestamp": current_date
#     }

#     with open(absolute_path, 'w') as f:
#         json.dump(full_transcript, f, indent=2)
    
#     logger.info(f"Transcript saved: {absolute_path}")
#     return absolute_path




# async def cleanup_local_files(combined_video_path: str, transcript_path: str,
#                               video_path: Optional[str] = None,
#                               audio_path: Optional[str] = None) -> bool:
#     """
#     Delete local files after successful upload to Azure
    
#     Args:
#         combined_video_path: Path to combined video file
#         transcript_path: Path to transcript JSON file
#         video_path: Optional path to original video file
#         audio_path: Optional path to original audio file
    
#     Returns:
#         True if all files were deleted successfully, False otherwise
#     """
#     files_to_delete = [
#         combined_video_path,
#         transcript_path
#     ]
    
#     if video_path:
#         files_to_delete.append(video_path)
#     if audio_path:
#         files_to_delete.append(audio_path)
    
#     all_deleted = True
#     for file_path in files_to_delete:
#         if file_path and os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#                 logger.info(f"Deleted: {file_path}")
#             except Exception as e:
#                 logger.error(f"Failed to delete {file_path}: {e}")
#                 all_deleted = False
#         elif file_path:
#             logger.warning(f"File not found: {file_path}")
    
#     return all_deleted


# async def update_user_transcript(text: str):
#     """Update user transcript in real-time"""
#     transcript_data["user_transcript"] = text
#     transcript_data["conversation_history"].append({
#         "speaker": "user",
#         "text": text,
#         "timestamp": datetime.now().isoformat()
#     })
#     logger.info(f"User: {text}")

# async def update_agent_transcript(text: str):
#     """Update agent transcript in real-time"""
#     transcript_data["agent_transcript"] = text
#     transcript_data["conversation_history"].append({
#         "speaker": "agent",
#         "text": text,
#         "timestamp": datetime.now().isoformat()
#     })
#     logger.info(f"Agent: {text}")

# async def transcription_node(text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]:
#     """Process transcription with real-time updates"""
#     full_text = ""
#     async for delta in text:
#         cleaned_delta = delta.replace("😘", "")
#         full_text += cleaned_delta
#         # Update agent transcript in real-time
#         await update_agent_transcript(full_text)
#         yield cleaned_delta


# async def entrypoint(ctx: JobContext):
#     """Main entrypoint for the LiveKit agent"""
    
  
#     async def shutdown_callback():
#         """Cleanup on shutdown - exit immediately, daemon threads handle rest"""
#         logger.info("Agent session ending - uploads continue in background")
#         logger.info("Agent session ended - process will exit now")

#     ctx.add_shutdown_callback(shutdown_callback)
#     await ctx.connect()

#     agent = Agent(
#         instructions=get_instruction("Remote.ai", "Best AI company",
#                                    "be the best ai company",
#                                    "AI engineer", "Get best AI engineer"),
#         tools=[lookup_weather],
#     )

#     # Create session with real-time transcription support
#     session = AgentSession(
#         vad=silero.VAD.load(),
#         stt=deepgram.STT(model="nova-3"),
#         llm=openai.LLM(model="gpt-4o-mini"),
#         tts=openai.TTS(),
#         use_tts_aligned_transcript=True,
#     )

#     # elevenlabs/eleven_turbo_v2_5:cjVigY5qzO86Huf0OWal



#     # Shared timestamps and tasks for perfect video/audio sync
#     participant_recording_times = {}
#     participant_recording_tasks = {}  # Track recording tasks for each participant
#     participant_recording_files = {}  # Track video/audio filenames for each participant
#     participant_transcript_files = {}  # Track transcript files for each participant
#     participant_upload_in_progress = {}  # Track if upload is in progress to prevent duplicates
#     background_tasks = []  # Track all background tasks for cleanup on shutdown
    
#     async def upload_and_cleanup_workflow(participant_identity: str, combined_video_path: str, 
#                                           transcript_path: str):
#         """
#         Complete workflow: Upload video and transcript to Azure, save links, then cleanup local files
        
#         Args:
#             participant_identity: Identity of the participant
#             combined_video_path: Path to the combined video file
#             transcript_path: Path to the transcript JSON file
#         """
#         # Prevent duplicate uploads
#         if participant_identity in participant_upload_in_progress:
#             if participant_upload_in_progress[participant_identity]:
#                 logger.info(f"Upload already in progress for {participant_identity}, skipping...")
#                 return
        
#         participant_upload_in_progress[participant_identity] = True
        
#         try:
#             logger.info(f"Starting upload and cleanup workflow for {participant_identity}")
            
#             # Step 1: Upload combined video to Azure
#             logger.info(f"Uploading combined video to Azure...")
#             video_upload_result = upload_video_to_azure(combined_video_path)
            
#             if not video_upload_result.get('success'):
#                 logger.error(f"Failed to upload video: {video_upload_result.get('error')}")
#                 return
            
#             video_url = video_upload_result['blob_url']
#             logger.info(f"Video uploaded successfully: {video_url}")
            
#             # Step 2: Upload transcript to Azure
#             logger.info(f"Uploading transcript to Azure...")
#             transcript_upload_result = upload_transcript_to_azure(transcript_path)
            
#             if not transcript_upload_result.get('success'):
#                 logger.error(f"Failed to upload transcript: {transcript_upload_result.get('error')}")
#                 return
            
#             transcript_url = transcript_upload_result['blob_url']
#             logger.info(f"Transcript uploaded successfully: {transcript_url}")
            
#             # Step 3: Save Azure links to file
#             logger.info(f"Saving Azure links...")
#             links_filename = None
#             try:
#                 links_result = save_azure_links(participant_identity, video_url, transcript_url, auto_evaluate=False)
#                 if links_result and isinstance(links_result, dict) and links_result.get('links_file'):
#                     links_filename = links_result.get('links_file')
#                     logger.info(f"Links saved successfully: {links_filename}")
#                 else:
#                     logger.warning(f"Links result format unexpected: {links_result}")
#                     # Fallback: construct filename manually
#                     links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
#                     logger.info(f"Using fallback links filename: {links_filename}")
#             except Exception as e:
#                 logger.error(f"Error saving links: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 # Still try to construct filename for evaluation
#                 links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
#                 logger.info(f"Using fallback links filename after error: {links_filename}")
            
#             # Step 3.5: Evaluation happens in disconnect handler, skip duplicate evaluation here
#             logger.info(f"Evaluation will run automatically in disconnect handler")
            
#             # Step 4: Get original video and audio paths for cleanup
#             original_video_path = None
#             original_audio_path = None
#             if participant_identity in participant_recording_files:
#                 original_video_path = participant_recording_files[participant_identity].get('video')
#                 original_audio_path = participant_recording_files[participant_identity].get('audio')
            
#             # Step 5: Cleanup local files
#             logger.info(f"🧹 Cleaning up local files...")
#             cleanup_success = await cleanup_local_files(
#                 combined_video_path=combined_video_path,
#                 transcript_path=transcript_path,
#                 video_path=original_video_path,
#                 audio_path=original_audio_path
#             )
            
#             if cleanup_success:
#                 logger.info(f"All local files cleaned up successfully!")
#             else:
#                 logger.warning(f"Some files could not be deleted")
            
#             logger.info(f"Upload and cleanup workflow completed for {participant_identity}")
            
#         except Exception as e:
#             logger.error(f"Error in upload and cleanup workflow: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             # Reset upload flag
#             participant_upload_in_progress[participant_identity] = False
    
#     async def combine_video_audio(participant_identity: str):
#         """Combine video and audio files into a single MP4 with perfect sync"""
#         if participant_identity not in participant_recording_files:
#             logger.warning(f"No recording files found for {participant_identity}")
#             return
        
#         files = participant_recording_files[participant_identity]
#         video_file = files.get('video')
#         audio_file = files.get('audio')
        
#         if not video_file or not audio_file:
#             logger.warning(f"Missing video or audio file for {participant_identity}")
#             logger.info(f"Video: {video_file}")
#             logger.info(f"Audio: {audio_file}")
#             return
        
#         # Check if both files exist
#         if not os.path.exists(video_file) or not os.path.exists(audio_file):
#             logger.error(f"Files not found:")
#             logger.error(f"Video: {os.path.exists(video_file)} - {video_file}")
#             logger.error(f"Audio: {os.path.exists(audio_file)} - {audio_file}")
#             return
        
#         # Create combined filename
#         os.makedirs("recordings/combined", exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         combined_file = f"recordings/combined/{participant_identity}_{timestamp}_COMBINED.mp4"
        
#         logger.info(f"Combining video and audio...")
#         logger.info(f"Video: {video_file}")
#         logger.info(f"Audio: {audio_file}")
#         logger.info(f"Output: {combined_file}")
        
#         try:
#             # Use ffmpeg to combine video and audio
#             # -c:v libx264 = H.264 codec (browser-compatible)
#             # -preset ultrafast = fastest encoding (for real-time)
#             # -c:a aac = encode audio as AAC
#             # -shortest = finish when shortest stream ends
#             cmd = [
#                 'ffmpeg',
#                 '-i', video_file,
#                 '-i', audio_file,
#                 '-c:v', 'libx264',       # H.264 for browser compatibility
#                 '-preset', 'ultrafast',   # Fast encoding
#                 '-c:a', 'aac',
#                 '-shortest',
#                 '-y',  # Overwrite output file if exists
#                 combined_file
#             ]
            
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 check=False
#             )
            
#             if result.returncode == 0:
#                 logger.info(f"Successfully combined video and audio!")
#                 logger.info(f"Combined file: {combined_file}")
                
#                 # Check file size
#                 size_mb = os.path.getsize(combined_file) / (1024 * 1024)
#                 logger.info(f"File size: {size_mb:.2f} MB")
                
#                 # Track combined file for this participant
#                 if participant_identity not in participant_recording_files:
#                     participant_recording_files[participant_identity] = {}
#                 participant_recording_files[participant_identity]['combined'] = combined_file
#                 logger.info(f"Video tracked and ready for upload (will be uploaded by disconnect handler)")
#             else:
#                 logger.error(f"ffmpeg error:")
#                 logger.error(f"{result.stderr}")
                
#         except FileNotFoundError:
#             logger.error(f"ffmpeg not found! Install with: brew install ffmpeg")
#         except Exception as e:
#             logger.error(f"Error combining files: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Record video with TIMESTAMP TRACKING for perfect audio sync
#     async def handle_video_recording(video_track: rtc.Track, participant_identity: str):
#         """Record video with timestamp tracking - fills gaps for perfect sync with audio"""
#         video_stream = rtc.VideoStream(video_track)
#         frames_with_timestamps = []  # Store (timestamp, frame) tuples
#         start_time = None
#         end_time = None
#         width = None
#         height = None
#         frame_count = 0
#         valid_frame_count = 0
        
#         try:
#             logger.info(f"Starting VIDEO recording for {participant_identity}")
#             logger.info(f"Mode: TIMESTAMP TRACKING (for perfect audio sync)")
            
#             # STEP 1: Collect ALL frames with timestamps
#             async for event in video_stream:
#                 try:
#                     video_frame: rtc.VideoFrame = event.frame
#                     frame_count += 1
#                     current_time = datetime.now()
                    
#                     # Use SHARED start time for this participant
#                     if participant_identity not in participant_recording_times:
#                         participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
#                         logger.info(f"Recording START time set: {current_time}")
                    
#                     if start_time is None:
#                         start_time = participant_recording_times[participant_identity]['start']
                    
#                     # Convert frame for recording (RGBA → BGR)
#                     frame_rgba = video_frame.convert(rtc.VideoBufferType.RGBA)
#                     frame_width = 1280
#                     frame_height = 720
                    
#                     # Skip frames with invalid dimensions (corrupted frames)
#                     if frame_width < 100 or frame_height < 100:
#                         logger.debug(f"Skipping corrupted frame {frame_count}: {frame_width}x{frame_height}")
#                         continue
                    
#                     # Capture width/height from FIRST valid frame only
#                     if width is None:
#                         width = frame_width
#                         height = frame_height
#                         logger.info(f"Detected resolution: {width}x{height}")
                    
#                     # Resize frames to match target resolution instead of skipping
#                     buffer = frame_rgba.data
#                     img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape((frame_height, frame_width, 4))
                    
#                     # Resize if resolution changed
#                     if frame_width != width or frame_height != height:
#                         logger.debug(f"Resizing frame {frame_count} from {frame_width}x{frame_height} to {width}x{height}")
#                         img_rgba = cv2.resize(img_rgba, (width, height), interpolation=cv2.INTER_LINEAR)
                    
#                     img_array = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                    
#                     # Store frame with its timestamp
#                     frames_with_timestamps.append((current_time, img_array.copy()))
#                     valid_frame_count += 1
                    
#                     # Log progress every 30 frames
#                     if valid_frame_count % 30 == 0:
#                         elapsed = (current_time - start_time).total_seconds()
#                         logger.info(f"Recorded {valid_frame_count} valid frames ({elapsed:.1f}s)")
                
#                 except Exception as frame_error:
#                     logger.error(f"Error processing frame {frame_count}: {frame_error}")
#                     continue
                    
#         except asyncio.CancelledError:
#             logger.info(f"Video recording cancelled (participant disconnected)")
#             # Use SHARED end time set by disconnect handler
#         except Exception as e:
#             logger.error(f"Video stream error: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             await video_stream.aclose()
            
#             # Use SHARED end time if set, otherwise use now
#             if participant_identity in participant_recording_times:
#                 end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
#             else:
#                 end_time = datetime.now()
            
#             # Update SHARED end time
#             if participant_identity in participant_recording_times:
#                 participant_recording_times[participant_identity]['end'] = end_time
#                 logger.info(f"Video recording END time set: {end_time}")
            
#             # STEP 2: Create video with EXACT duration matching audio (fill gaps with duplicate frames)
#             if start_time and end_time and len(frames_with_timestamps) > 0 and width and height:
#                 # Use SHARED timestamps if available (both tracks might still be recording)
#                 if participant_identity in participant_recording_times:
#                     shared_start = participant_recording_times[participant_identity]['start']
#                     shared_end = participant_recording_times[participant_identity]['end'] or end_time
#                     start_time = shared_start
#                     end_time = shared_end
                
#                 total_duration = (end_time - start_time).total_seconds()
#                 target_fps = 30.0  # Target FPS for smooth video
                
#                 logger.info(f"Frame collection complete!")
#                 logger.info(f"Valid frames collected: {len(frames_with_timestamps)}")
#                 logger.info(f"Total recording duration: {total_duration:.1f} seconds")
#                 logger.info(f"Video resolution: {width}x{height}")
                
#                 # Validate resolution
#                 if width < 100 or height < 100:
#                     logger.error(f"Invalid resolution: {width}x{height}")
#                     return
                
#                 # STEP 3: Generate frames at regular intervals (30 FPS) to match audio duration
#                 target_frame_count = int(total_duration * target_fps)
#                 logger.info(f"Target frames for {total_duration:.1f}s @ {target_fps} FPS: {target_frame_count}")
                
#                 output_frames = []
#                 frame_idx = 0
                
#                 for i in range(target_frame_count):
#                     # Calculate target time for this frame
#                     target_time = start_time + (i / target_fps) * timedelta(seconds=1)
                    
#                     # Find closest actual frame to this target time
#                     while frame_idx < len(frames_with_timestamps) - 1:
#                         current_frame_time = frames_with_timestamps[frame_idx][0]
#                         next_frame_time = frames_with_timestamps[frame_idx + 1][0]
                        
#                         # If next frame is closer to target, move forward
#                         if abs((next_frame_time - target_time).total_seconds()) < abs((current_frame_time - target_time).total_seconds()):
#                             frame_idx += 1
#                         else:
#                             break
                    
#                     # Use the closest frame
#                     output_frames.append(frames_with_timestamps[frame_idx][1])
                
#                 logger.info(f"Generated {len(output_frames)} frames (filled gaps for continuous playback)")
                
#                 # STEP 4: Write ALL frames to video
#                 os.makedirs("recordings/video", exist_ok=True)
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 video_filename = f"recordings/video/{participant_identity}_{timestamp}_VIDEO.mp4"
                
#                 # Use H.264 codec (avc1) for browser compatibility
#                 fourcc = cv2.VideoWriter_fourcc(*'avc1')
#                 video_writer = cv2.VideoWriter(video_filename, fourcc, target_fps, (width, height))
                
#                 if not video_writer.isOpened():
#                     logger.error(f"Failed to open video writer!")
#                     return
                
#                 logger.info(f"Writing {len(output_frames)} frames to: {video_filename}")
#                 logger.info(f"Settings: {width}x{height} @ {target_fps} FPS")
                
#                 # Write ALL frames
#                 for i, frame in enumerate(output_frames):
#                     video_writer.write(frame)
#                     if (i + 1) % 100 == 0 or (i + 1) == len(output_frames):
#                         logger.info(f"Written {i + 1}/{len(output_frames)} frames...")
                
#                 video_writer.release()
#                 logger.info(f"Video writer released")
                
#                 logger.info(f"Video saved successfully!")
#                 logger.info(f"File: {video_filename}")
#                 logger.info(f"Actual frames captured: {len(frames_with_timestamps)}")
#                 logger.info(f"Output frames (with gap filling): {len(output_frames)}")
#                 logger.info(f"Video duration: {total_duration:.1f} seconds")
#                 logger.info(f"FPS: {target_fps}")
#                 logger.info(f"Gap filling: {len(output_frames) - len(frames_with_timestamps)} duplicate frames")
#                 logger.info(f"Video duration will MATCH audio duration perfectly!")
                
#                 # Track video filename for combining later
#                 if participant_identity not in participant_recording_files:
#                     participant_recording_files[participant_identity] = {}
#                 participant_recording_files[participant_identity]['video'] = video_filename
#                 logger.info(f"Video filename tracked for combining")
                
#                 # Check if audio is also done - if yes, combine them!
#                 if 'audio' in participant_recording_files.get(participant_identity, {}):
#                     logger.info(f"Both video and audio ready - combining now!")
#                     task = asyncio.create_task(combine_video_audio(participant_identity))
#                     background_tasks.append(task)
#             else:
#                 logger.warning(f"No valid frames were recorded!")
    
#     # Record audio with SHARED TIMESTAMPS for perfect video sync
#     async def handle_audio_recording(audio_track: rtc.Track, participant_identity: str):
#         """Record audio with shared timestamps - matches video duration"""
#         audio_stream = rtc.AudioStream(audio_track)
#         audio_frames = []
#         audio_filename = None
#         sample_rate = None
#         num_channels = None
#         audio_frame_count = 0
#         start_time = None
        
#         try:
#             logger.info(f"Starting AUDIO recording for {participant_identity}")
            
#             async for event in audio_stream:
#                 current_time = datetime.now()
#                 audio_frame: rtc.AudioFrame = event.frame
#                 audio_frame_count += 1
                
#                 # Use SHARED start time for this participant
#                 if participant_identity not in participant_recording_times:
#                     participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
#                     logger.info(f"Recording START time set: {current_time}")
                
#                 if start_time is None:
#                     start_time = participant_recording_times[participant_identity]['start']
                
#                 # Get audio properties from first frame
#                 if sample_rate is None:
#                     sample_rate = audio_frame.sample_rate
#                     num_channels = audio_frame.num_channels
                    
#                     os.makedirs("recordings/audio", exist_ok=True)
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     audio_filename = f"recordings/audio/{participant_identity}_{timestamp}_AUDIO.wav"
                    
#                     logger.info(f"Recording to: {audio_filename}")
#                     logger.info(f"Sample rate: {sample_rate}Hz, Channels: {num_channels}")
                
#                 # Store audio data (no processing, keep quality)
#                 audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
#                 audio_frames.append(audio_data.tobytes())
                
#                 # Log progress
#                 if audio_frame_count % 100 == 0:
#                     logger.info(f"Audio: {audio_frame_count} frames")
                    
#         except asyncio.CancelledError:
#             logger.info(f"Audio recording cancelled (participant disconnected)")
#             # Use SHARED end time set by disconnect handler
#         except Exception as e:
#             logger.error(f"Audio error: {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             await audio_stream.aclose()
            
#             # Use SHARED end time if set, otherwise use now
#             if participant_identity in participant_recording_times:
#                 end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
#             else:
#                 end_time = datetime.now()
            
#             # Update SHARED end time
#             if participant_identity in participant_recording_times:
#                 # Use the LATEST end time between video and audio
#                 current_end = participant_recording_times[participant_identity]['end']
#                 if current_end is None or end_time > current_end:
#                     participant_recording_times[participant_identity]['end'] = end_time
#                 logger.info(f"Audio recording END time set: {end_time}")
            
#             # Save audio to WAV (lossless)
#             if audio_frames and sample_rate:
#                 try:
#                     with wave.open(audio_filename, 'wb') as wav_file:
#                         wav_file.setnchannels(num_channels)
#                         wav_file.setsampwidth(2)  # 16-bit
#                         wav_file.setframerate(sample_rate)
#                         wav_file.writeframes(b''.join(audio_frames))
                    
#                     # Log with shared timestamps
#                     if participant_identity in participant_recording_times:
#                         shared_start = participant_recording_times[participant_identity]['start']
#                         shared_end = participant_recording_times[participant_identity]['end']
#                         if shared_start and shared_end:
#                             duration = (shared_end - shared_start).total_seconds()
#                             logger.info(f"Audio saved: {audio_filename} ({audio_frame_count} frames)")
#                             logger.info(f"Audio uses SHARED timeline: {duration:.1f} seconds")
                    
#                     # Track audio filename for combining later
#                     if participant_identity not in participant_recording_files:
#                         participant_recording_files[participant_identity] = {}
#                     participant_recording_files[participant_identity]['audio'] = audio_filename
#                     logger.info(f"Audio filename tracked for combining")
                    
#                     # Check if video is also done - if yes, combine them!
#                     if 'video' in participant_recording_files.get(participant_identity, {}):
#                         logger.info(f"Both video and audio ready - combining now!")
#                         task = asyncio.create_task(combine_video_audio(participant_identity))
#                         background_tasks.append(task)
                    
#                 except Exception as e:
#                     logger.error(f"Error saving audio: {e}")
#                     import traceback
#                     traceback.print_exc()

#     # Participant disconnect handler - ensures both video and audio stop together
#     @ctx.room.on("participant_disconnected")
#     def on_participant_disconnected(participant: rtc.RemoteParticipant):
#         """Handle participant disconnect - IMMEDIATELY stop both video and audio"""
#         disconnect_time = datetime.now()
#         logger.info(f"Participant disconnected: {participant.identity}")
        
#         # Set SHARED end time IMMEDIATELY
#         if participant.identity in participant_recording_times:
#             participant_recording_times[participant.identity]['end'] = disconnect_time
#             logger.info(f"SHARED END time locked: {disconnect_time}")
            
#             # Calculate final duration
#             start = participant_recording_times[participant.identity]['start']
#             duration = (disconnect_time - start).total_seconds()
#             logger.info(f"Final recording duration: {duration:.1f} seconds")
#             logger.info(f"Both video and audio will use this EXACT timeline")
        
#         # Cancel recording tasks to stop immediately (not wait for streams)
#         if participant.identity in participant_recording_tasks:
#             tasks = participant_recording_tasks[participant.identity]
#             for task_name, task in tasks.items():
#                 if task and not task.done():
#                     task.cancel()
#                     logger.info(f"Cancelled {task_name} recording task")
#             logger.info(f"Both recordings stopped at SAME TIME: {disconnect_time}")
        
#         # Save transcript, upload to Azure, and save links - then evaluation runs from Azure
#         async def save_and_upload_workflow():
#             """Save transcript, upload to Azure, save links, then evaluate from Azure"""
#             try:
#                 logger.info(f"Saving transcript for {participant.identity}...")
#                 transcript_path = await write_transcript(session)
                
#                 if transcript_path:
#                     participant_transcript_files[participant.identity] = transcript_path
#                     logger.info(f"Transcript saved: {transcript_path}")
                    
#                     # Wait for video to be ready (with timeout)
#                     logger.info(f"⏳ Waiting for video to be ready...")
#                     max_wait = 15  # seconds
#                     waited = 0
#                     combined_video_path = None
                    
#                     while waited < max_wait:
#                         if participant.identity in participant_recording_files:
#                             combined_video_path = participant_recording_files[participant.identity].get('combined')
#                             if combined_video_path and os.path.exists(combined_video_path):
#                                 logger.info(f"Video ready: {combined_video_path}")
#                                 break
#                         await asyncio.sleep(1)
#                         waited += 1
                    
#                     # Upload to Azure in background thread (daemon - won't block exit)
#                     def upload_and_save_links_background():
#                         """Upload to Azure, save links JSON, trigger API evaluation"""
#                         try:
#                             import time
#                             time.sleep(0.5)  # Brief delay
                            
#                             from azure_storage import upload_video_to_azure, upload_transcript_to_azure
#                             from save_links import save_azure_links
                            
#                             # Upload transcript to Azure
#                             logger.info(f"Uploading transcript to Azure...")
#                             transcript_result = upload_transcript_to_azure(transcript_path)
                            
#                             if not transcript_result.get('success'):
#                                 logger.error(f"Failed to upload transcript: {transcript_result.get('error')}")
#                                 return
                            
#                             transcript_url = transcript_result['blob_url']
#                             logger.info(f"Transcript uploaded: {transcript_url[:80]}...")
                            
#                             # Upload video if ready
#                             video_url = None
#                             if combined_video_path and os.path.exists(combined_video_path):
#                                 logger.info(f"Uploading video to Azure...")
#                                 video_result = upload_video_to_azure(combined_video_path)
                                
#                                 if video_result.get('success'):
#                                     video_url = video_result['blob_url']
#                                     logger.info(f"Video uploaded: {video_url[:80]}...")
#                                 else:
#                                     logger.warning(f"Video upload failed: {video_result.get('error')}")
#                                     video_url = "pending"
#                             else:
#                                 logger.warning(f"Video not ready yet, using placeholder")
#                                 video_url = "pending"
                            
#                             # Save links JSON (fast, no evaluation here)
#                             logger.info(f"Saving Azure links to JSON file...")
#                             result = save_azure_links(
#                                 participant_identity=participant.identity,
#                                 video_url=video_url,
#                                 transcript_url=transcript_url,
#                                 auto_evaluate=False  # DON'T evaluate in agent - too slow
#                             )
                            
#                             if result and result.get('success'):
#                                 links_file = result.get('links_file')
#                                 logger.info(f"Links saved: {links_file}")
                                
#                                 # Trigger evaluation via API (fire and forget)
#                                 try:
#                                     import requests
#                                     api_url = os.getenv('API_SERVER_URL', 'http://localhost:5001')
#                                     logger.info(f"Triggering evaluation via API...")
                                    
#                                     requests.post(
#                                         f"{api_url}/api/evaluations/evaluate_from_links",
#                                         json={
#                                             "links_file": links_file,
#                                             "participant_identity": participant.identity
#                                         },
#                                         timeout=2
#                                     )
#                                     logger.info(f"Evaluation API triggered")
#                                 except Exception as e:
#                                     logger.warning(f"Failed to trigger evaluation API: {e}")
#                             else:
#                                 logger.error(f"Failed to save links")
                                
#                         except Exception as e:
#                             logger.error(f"Upload error: {e}")
#                             import traceback
#                             traceback.print_exc()
                    
#                     # Start upload in daemon thread (won't block exit, NOT tracked in background_tasks)
#                     upload_thread = threading.Thread(target=upload_and_save_links_background, daemon=True, name="UploadThread")
#                     upload_thread.start()
#                     logger.info(f"Upload started in background (independent daemon thread)")
                    
#                 else:
#                     logger.error(f"Failed to save transcript")
#             except Exception as e:
#                 logger.error(f"Error: {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         # Start transcript save, upload, and evaluation workflow
#         task = asyncio.create_task(save_and_upload_workflow())
#         background_tasks.append(task)
    
#     # Set up track subscription handler - NO SYNC, separate recordings
#     @ctx.room.on("track_subscribed")
#     def on_track_subscribed(
#         track: rtc.Track,
#         publication: rtc.TrackPublication,
#         participant: rtc.RemoteParticipant,
#     ):
#         """Handle track subscriptions - record video and audio SEPARATELY"""
#         logger.info(f"Track subscribed from {participant.identity}: {track.kind}")
        
#         # Initialize task tracking for this participant
#         if participant.identity not in participant_recording_tasks:
#             participant_recording_tasks[participant.identity] = {}
        
#         if track.kind == rtc.TrackKind.KIND_VIDEO:
#             # Start video recording immediately - NO WAIT for audio
#             logger.info(f"📹 Video track - starting independent recording")
#             video_task = asyncio.create_task(handle_video_recording(track, participant.identity))
#             participant_recording_tasks[participant.identity]['video'] = video_task
        
#         elif track.kind == rtc.TrackKind.KIND_AUDIO:
#             # Start audio recording immediately - NO WAIT for video
#             logger.info(f"Audio track - starting independent recording")
#             audio_task = asyncio.create_task(handle_audio_recording(track, participant.identity))
#             participant_recording_tasks[participant.identity]['audio'] = audio_task

#     logger.info("Starting agent session...")
#     await session.start(agent=agent, room=ctx.room)

  

#     # Generate initial greeting
#     await session.generate_reply(instructions="greet the user and ask about their day")
#     logger.info("Agent session started successfully")


# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))



# from prompt import get_instruction
# from datetime import datetime, timedelta
# import json
# import asyncio
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     JobContext,
#     RunContext,
#     WorkerOptions,
#     cli,
#     function_tool,
#     utils,
# )
# from livekit.plugins import deepgram, elevenlabs, openai, silero
# from dotenv import load_dotenv
# import os
# from livekit.agents import AgentSession, Agent, RoomInputOptions, inference
# from livekit.agents import ModelSettings
# from livekit import rtc
# from typing import AsyncIterable, Optional
# import logging
# import cv2
# import numpy as np
# import wave
# from combine_video_audio import combine_video_audio
# from upload_and_cleanup import start_upload_in_background
# import threading
# #
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Suppress ALL Deepgram errors - they're expected during disconnect
# class DeepgramErrorFilter(logging.Filter):
#     """Filter to suppress ALL Deepgram errors"""
#     def filter(self, record):
#         # Suppress all errors from Deepgram-related loggers
#         logger_name = record.name.lower()
#         if "deepgram"in logger_name or "livekit.plugins.deepgram"in logger_name:
#             return False  # Don't log any Deepgram errors
#         return True  # Log everything else
#
#
#
#
# async def evaluate_interview(
#         links_file_path,
#         job_description=None,
#         candidate_cv=None,
#         candidate_info=None,
#         evaluation_instruction=None,
#         use_api=False
# ):
#     from evaluate import evaluate_interview_comprehensive
#     from prompt import load_transcript_from_file
#
#     interview_transcript = load_transcript_from_file(links_file_path)
#
#     # Step 4: Load job description, CV, and candidate info if not provided
#     participant_identity = None
#     if not job_description or not candidate_cv or not candidate_info:
#
#         filename = os.path.basename(links_file_path)
#         # Remove _links.json suffix
#         if filename.endswith('_links.json'):
#             name_without_suffix = filename[:-11]  # Remove '_links.json'
#             # Remove timestamp (last two parts: YYYYMMDD_HHMMSS)
#             parts = name_without_suffix.split('_')
#             if len(parts) >= 3:
#                 # Take all parts except the last two (date and time)
#                 participant_identity = '_'.join(parts[:-2])
#             else:
#                 participant_identity = name_without_suffix
#         else:
#             # Fallback: just remove .json if present
#             participant_identity = filename.replace('.json', '').replace('_links', '')
#
#         # Try participant-specific config first
#         participant_config_file = f"config_{participant_identity}.json"
#         config_file = None
#
#         if os.path.exists(participant_config_file):
#             config_file = participant_config_file
#         elif os.path.exists("config.json"):
#             config_file = "config.json"
#
#         if config_file:
#             with open(config_file, 'r') as f:
#                 config = json.load(f)
#                 if not job_description:
#                     job_description = config.get("job_description", {})
#                 if not candidate_cv:
#                     candidate_cv = config.get("candidate_cv", {})
#                 if not candidate_info:
#                     candidate_info = config.get("candidate_info", {})
#
#     # Use defaults if still empty
#     if not candidate_info:
#         participant_id = participant_identity if participant_identity else "unknown"
#         candidate_info = {
#             "candidate_id": participant_id,
#             "candidate_email": f"{participant_id}@example.com",
#             "candidate_name": participant_id.replace("_", " ").title(),
#             "job_id": os.getenv("JOB_ID", "default_job")
#         }
#
#     if not job_description:
#         job_description = {
#             "title": os.getenv("JOB_TITLE", "AI Engineer"),
#             "department": os.getenv("JOB_DEPARTMENT", "Technology"),
#             "location": os.getenv("JOB_LOCATION", "Remote"),
#             "employment_type": os.getenv("JOB_EMPLOYMENT_TYPE", "Full-time"),
#             "summary": os.getenv("JOB_SUMMARY", "We are seeking a qualified candidate for this position."),
#             "responsibilities": [],
#             "requirements": [],
#             "skills": []
#         }
#
#     evaluation = None
#
#     evaluation = evaluate_interview_comprehensive(
#         job_description=job_description,
#         candidate_cv=candidate_cv or {},
#         candidate_info=candidate_info,
#         interview_transcript=interview_transcript,
#         evaluation_instruction=evaluation_instruction,
#         use_api=use_api
#     )
#
#     return f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10", f"Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10"
#
#
#
#
# # Apply filter to suppress all Deepgram errors
# for logger_name in ["livekit.plugins.deepgram.stt", "livekit.plugins.deepgram", "livekit.agents.stt", "livekit.agents"]:
#     log = logging.getLogger(logger_name)
#     log.addFilter(DeepgramErrorFilter())
#
# load_dotenv(".env")
#
# # Global transcript storage for real-time updates
#
# @function_tool
# async def lookup_weather(
#     context: RunContext,
#     location: str,
# ):
#     """Used to look up weather information."""
#     return {"weather": "sunny", "temperature": 70}
#
#
# transcript_data = {
#     "user_transcript": "",
#     "agent_transcript": "",
#     "conversation_history": []
# }
#
#
# async def write_transcript(session: AgentSession) -> Optional[str]:
#     """Save transcript to file and return the absolute path"""
#     current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_directory = "."
#     filename = os.path.join(save_directory, f"transcript_{current_date}.json")
#     # Convert to absolute path
#     absolute_path = os.path.abspath(filename)
#
#     # Save both session history and our real-time transcript data
#     full_transcript = {
#         "session_history": session.history.to_dict(),
#         "real_time_transcript": transcript_data,
#         "timestamp": current_date
#     }
#
#     with open(absolute_path, 'w') as f:
#         json.dump(full_transcript, f, indent=2)
#
#     return absolute_path
#
#
# async def update_user_transcript(text: str):
#     """Update user transcript in real-time"""
#     transcript_data["user_transcript"] = text
#     transcript_data["conversation_history"].append({
#         "speaker": "user",
#         "text": text,
#         "timestamp": datetime.now().isoformat()
#     })
#
#
#
# async def update_agent_transcript(text: str):
#     """Update agent transcript in real-time"""
#     transcript_data["agent_transcript"] = text
#     transcript_data["conversation_history"].append({
#         "speaker": "agent",
#         "text": text,
#         "timestamp": datetime.now().isoformat()
#     })
#
# async def transcription_node(text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]:
#     """Process transcription with real-time updates"""
#     full_text = ""
#     async for delta in text:
#         cleaned_delta = delta.replace("😘", "")
#         full_text += cleaned_delta
#         # Update agent transcript in real-time
#         await update_agent_transcript(full_text)
#         yield cleaned_delta
#
#
# async def entrypoint(ctx: JobContext):
#     """Main entrypoint for the LiveKit agent"""
#
#
#     async def shutdown_callback():
#         """Cleanup on shutdown - exit immediately, daemon threads handle rest"""
#         try:
#             # Gracefully close session to avoid Deepgram connection errors
#             if session:
#                 await asyncio.sleep(0.1)  # Brief delay to allow cleanup
#         except Exception:
#             pass
#
#     ctx.add_shutdown_callback(shutdown_callback)
#     await ctx.connect()
#
#     agent = Agent(
#         instructions=get_instruction("Remote.ai", "Best AI company",
#                                    "be the best ai company",
#                                    "AI engineer", "Get best AI engineer"),
#         tools=[lookup_weather],
#     )
#
#     # Create session with real-time transcription support
#     session = AgentSession(
#         vad=silero.VAD.load(),
#         stt=deepgram.STT(model="nova-3"),
#         llm=openai.LLM(model="gpt-4o-mini"),
#         tts=openai.TTS(),
#         use_tts_aligned_transcript=True,
#     )
#
#
#
#
#     # Shared timestamps and tasks for perfect video/audio sync
#     participant_recording_times = {}
#     participant_recording_tasks = {}  # Track recording tasks for each participant
#     participant_recording_files = {}  # Track video/audio filenames for each participant
#     participant_transcript_files = {}  # Track transcript files for each participant
#     participant_upload_in_progress = {}  # Track if upload is in progress to prevent duplicates
#     participant_evaluation_results = {}  # Track evaluation results for each participant
#     background_tasks = []  # Track all background tasks for cleanup on shutdown
#
#
#
#
#     # Record video with TIMESTAMP TRACKING for perfect audio sync
#     async def handle_video_recording(video_track: rtc.Track, participant_identity: str):
#         """Record video with timestamp tracking - fills gaps for perfect sync with audio"""
#         video_stream = rtc.VideoStream(video_track)
#         frames_with_timestamps = []  # Store (timestamp, frame) tuples
#         start_time = None
#         end_time = None
#         width = None
#         height = None
#         frame_count = 0
#         valid_frame_count = 0
#
#         try:
#             # STEP 1: Collect ALL frames with timestamps
#             async for event in video_stream:
#                 try:
#                     video_frame: rtc.VideoFrame = event.frame
#                     frame_count += 1
#                     current_time = datetime.now()
#
#                     # Use SHARED start time for this participant
#                     if participant_identity not in participant_recording_times:
#                         participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
#
#                     if start_time is None:
#                         start_time = participant_recording_times[participant_identity]['start']
#
#                     # Convert frame for recording (RGBA → BGR)
#                     frame_rgba = video_frame.convert(rtc.VideoBufferType.RGBA)
#                     frame_width = frame_rgba.width
#                     frame_height = frame_rgba.height
#
#                     # Skip frames with invalid dimensions (corrupted frames)
#                     if frame_width < 100 or frame_height < 100:
#                         continue
#
#                     # Capture width/height from FIRST valid frame only
#                     if width is None:
#                         width = frame_width
#                         height = frame_height
#
#                     # Resize frames to match target resolution instead of skipping
#                     buffer = frame_rgba.data
#                     img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape((frame_height, frame_width, 4))
#
#                     # Resize if resolution changed
#                     if frame_width != width or frame_height != height:
#                         img_rgba = cv2.resize(img_rgba, (width, height), interpolation=cv2.INTER_LINEAR)
#
#                     img_array = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
#
#                     # Store frame with its timestamp
#                     frames_with_timestamps.append((current_time, img_array.copy()))
#                     valid_frame_count += 1
#
#                 except Exception:
#                     continue
#
#         except asyncio.CancelledError:
#             pass
#         except Exception:
#             pass
#         finally:
#             await video_stream.aclose()
#
#             # Use SHARED end time if set, otherwise use now
#             if participant_identity in participant_recording_times:
#                 end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
#             else:
#                 end_time = datetime.now()
#
#             # Update SHARED end time
#             if participant_identity in participant_recording_times:
#                 participant_recording_times[participant_identity]['end'] = end_time
#
#             # STEP 2: Create video with EXACT duration matching audio (fill gaps with duplicate frames)
#             if start_time and end_time and len(frames_with_timestamps) > 0 and width and height:
#                 # Use SHARED timestamps if available (both tracks might still be recording)
#                 if participant_identity in participant_recording_times:
#                     shared_start = participant_recording_times[participant_identity]['start']
#                     shared_end = participant_recording_times[participant_identity]['end'] or end_time
#                     start_time = shared_start
#                     end_time = shared_end
#
#                 total_duration = (end_time - start_time).total_seconds()
#                 target_fps = 30.0  # Target FPS for smooth video
#
#                 # Validate resolution
#                 if width < 100 or height < 100:
#                     return
#
#                 # STEP 3: Generate frames at regular intervals (30 FPS) to match audio duration
#                 target_frame_count = int(total_duration * target_fps)
#
#                 output_frames = []
#                 frame_idx = 0
#
#                 for i in range(target_frame_count):
#                     # Calculate target time for this frame
#                     target_time = start_time + (i / target_fps) * timedelta(seconds=1)
#
#                     # Find closest actual frame to this target time
#                     while frame_idx < len(frames_with_timestamps) - 1:
#                         current_frame_time = frames_with_timestamps[frame_idx][0]
#                         next_frame_time = frames_with_timestamps[frame_idx + 1][0]
#
#                         # If next frame is closer to target, move forward
#                         if abs((next_frame_time - target_time).total_seconds()) < abs((current_frame_time - target_time).total_seconds()):
#                             frame_idx += 1
#                         else:
#                             break
#
#                     # Use the closest frame
#                     output_frames.append(frames_with_timestamps[frame_idx][1])
#
#                 # STEP 4: Write ALL frames to video
#                 os.makedirs("recordings/video", exist_ok=True)
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 video_filename = f"recordings/video/{participant_identity}_{timestamp}_VIDEO.mp4"
#
#                 # Use H.264 codec (avc1) for browser compatibility
#                 fourcc = cv2.VideoWriter_fourcc(*'avc1')
#                 video_writer = cv2.VideoWriter(video_filename, fourcc, target_fps, (width, height))
#
#                 if not video_writer.isOpened():
#                     return
#
#                 # Write ALL frames
#                 for i, frame in enumerate(output_frames):
#                     video_writer.write(frame)
#
#                 video_writer.release()
#
#                 # Track video filename for combining later
#                 if participant_identity not in participant_recording_files:
#                     participant_recording_files[participant_identity] = {}
#                 participant_recording_files[participant_identity]['video'] = video_filename
#
#                 # Check if audio is also done - if yes, combine them!
#                 if 'audio' in participant_recording_files.get(participant_identity, {}):
#                     task = asyncio.create_task(combine_video_audio(participant_identity, participant_recording_files))
#                     background_tasks.append(task)
#
#     # Record audio with SHARED TIMESTAMPS for perfect video sync
#     async def handle_audio_recording(audio_track: rtc.Track, participant_identity: str):
#         """Record audio with shared timestamps - matches video duration"""
#         audio_stream = rtc.AudioStream(audio_track)
#         audio_frames = []
#         audio_filename = None
#         sample_rate = None
#         num_channels = None
#         audio_frame_count = 0
#         start_time = None
#
#         try:
#             async for event in audio_stream:
#                 current_time = datetime.now()
#                 audio_frame: rtc.AudioFrame = event.frame
#                 audio_frame_count += 1
#
#                 # Use SHARED start time for this participant
#                 if participant_identity not in participant_recording_times:
#                     participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
#
#                 if start_time is None:
#                     start_time = participant_recording_times[participant_identity]['start']
#
#                 # Get audio properties from first frame
#                 if sample_rate is None:
#                     sample_rate = audio_frame.sample_rate
#                     num_channels = audio_frame.num_channels
#
#                     os.makedirs("recordings/audio", exist_ok=True)
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     audio_filename = f"recordings/audio/{participant_identity}_{timestamp}_AUDIO.wav"
#
#                 # Store audio data (no processing, keep quality)
#                 audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
#                 audio_frames.append(audio_data.tobytes())
#
#         except asyncio.CancelledError:
#             pass
#         except Exception:
#             pass
#         finally:
#             await audio_stream.aclose()
#
#             # Use SHARED end time if set, otherwise use now
#             if participant_identity in participant_recording_times:
#                 end_time = participant_recording_times[participant_identity]['end'] or datetime.now()
#             else:
#                 end_time = datetime.now()
#
#             # Update SHARED end time
#             if participant_identity in participant_recording_times:
#                 # Use the LATEST end time between video and audio
#                 current_end = participant_recording_times[participant_identity]['end']
#                 if current_end is None or end_time > current_end:
#                     participant_recording_times[participant_identity]['end'] = end_time
#
#             # Save audio to WAV (lossless)
#             if audio_frames and sample_rate:
#                 try:
#                     with wave.open(audio_filename, 'wb') as wav_file:
#                         wav_file.setnchannels(num_channels)
#                         wav_file.setsampwidth(2)  # 16-bit
#                         wav_file.setframerate(sample_rate)
#                         wav_file.writeframes(b''.join(audio_frames))
#
#                     # Track audio filename for combining later
#                     if participant_identity not in participant_recording_files:
#                         participant_recording_files[participant_identity] = {}
#                     participant_recording_files[participant_identity]['audio'] = audio_filename
#
#                     # Check if video is also done - if yes, combine them!
#                     if 'video' in participant_recording_files.get(participant_identity, {}):
#                         task = asyncio.create_task(combine_video_audio(participant_identity, participant_recording_files))
#                         background_tasks.append(task)
#
#                 except Exception:
#                     pass
#
#     # Participant disconnect handler - ensures both video and audio stop together
#     @ctx.room.on("participant_disconnected")
#     def on_participant_disconnected(participant: rtc.RemoteParticipant):
#         """Handle participant disconnect - IMMEDIATELY stop both video and audio"""
#         disconnect_time = datetime.now()
#
#         # Set SHARED end time IMMEDIATELY
#         if participant.identity in participant_recording_times:
#             participant_recording_times[participant.identity]['end'] = disconnect_time
#
#         # Cancel recording tasks to stop immediately (not wait for streams)
#         if participant.identity in participant_recording_tasks:
#             tasks = participant_recording_tasks[participant.identity]
#             for task_name, task in tasks.items():
#                 if task and not task.done():
#                     task.cancel()
#
#
#         # Save transcript, then run evaluation and video upload in parallel
#         async def workflow():
#             try:
#                 transcript_path = await write_transcript(session)
#                 if not transcript_path:
#                     return
#
#                 participant_transcript_files[participant.identity] = transcript_path
#
#                 # Start evaluation in background
#                 async def eval_task():
#                     try:
#                         print("Evaluation started")
#                         evaluation_result = await evaluate_interview(transcript_path)
#                         participant_evaluation_results[participant.identity] = evaluation_result
#                     except Exception:
#                         participant_evaluation_results[participant.identity] = None
#
#                 # Wait for video and upload
#                 async def upload_task():
#                     try:
#                         max_wait = 15
#                         waited = 0
#                         combined_video_path = None
#
#                         while waited < max_wait:
#                             if participant.identity in participant_recording_files:
#                                 combined_video_path = participant_recording_files[participant.identity].get('combined')
#                                 if combined_video_path and os.path.exists(combined_video_path):
#                                     break
#                             await asyncio.sleep(1)
#                             waited += 1
#
#                         recording_files = participant_recording_files.get(participant.identity, {})
#                         video_path = recording_files.get('video')
#                         audio_path = recording_files.get('audio')
#
#                         start_upload_in_background(
#                             participant_identity=participant.identity,
#                             transcript_path=transcript_path,
#                             combined_video_path=combined_video_path,
#                             video_path=video_path,
#                             audio_path=audio_path,
#                             evaluation_results=participant_evaluation_results
#                         )
#                     except Exception:
#                         pass
#
#                 # Start both tasks in parallel
#                 background_tasks.append(asyncio.create_task(eval_task()))
#                 background_tasks.append(asyncio.create_task(upload_task()))
#
#             except Exception:
#                 pass
#
#         asyncio.create_task(workflow())
#
#     # Set up track subscription handler - NO SYNC, separate recordings
#     @ctx.room.on("track_subscribed")
#     def on_track_subscribed(
#         track: rtc.Track,
#         publication: rtc.TrackPublication,
#         participant: rtc.RemoteParticipant,
#     ):
#         """Handle track subscriptions - record video and audio SEPARATELY"""
#         # Initialize task tracking for this participant
#         if participant.identity not in participant_recording_tasks:
#             participant_recording_tasks[participant.identity] = {}
#
#         if track.kind == rtc.TrackKind.KIND_VIDEO:
#             # Start video recording immediately - NO WAIT for audio
#             video_task = asyncio.create_task(handle_video_recording(track, participant.identity))
#             participant_recording_tasks[participant.identity]['video'] = video_task
#
#         elif track.kind == rtc.TrackKind.KIND_AUDIO:
#             # Start audio recording immediately - NO WAIT for video
#             audio_task = asyncio.create_task(handle_audio_recording(track, participant.identity))
#             participant_recording_tasks[participant.identity]['audio'] = audio_task
#
#     try:
#         await session.start(agent=agent, room=ctx.room)
#         # Generate initial greeting
#         await session.generate_reply(instructions="greet the user")
#     except Exception:
#         pass
#
#
# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


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
        "timestamp": current_date
    }

    with open(absolute_path, 'w') as f:
        json.dump(full_transcript, f, indent=2)

    logger.info(f"Transcript saved: {absolute_path}")
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
            "job_id": os.getenv("JOB_ID", "default_job")
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

    return f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10", f"Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10"


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
                logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                all_deleted = False
        elif file_path:
            logger.warning(f"File not found: {file_path}")

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


async def update_agent_transcript(text: str):
    """Update agent transcript in real-time"""
    transcript_data["agent_transcript"] = text
    transcript_data["conversation_history"].append({
        "speaker": "agent",
        "text": text,
        "timestamp": datetime.now().isoformat()
    })
    logger.info(f"Agent: {text}")


async def transcription_node(text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[str]:
    """Process transcription with real-time updates"""
    full_text = ""
    async for delta in text:
        cleaned_delta = delta.replace("😘", "")
        full_text += cleaned_delta
        # Update agent transcript in real-time
        await update_agent_transcript(full_text)
        yield cleaned_delta


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent"""

    async def shutdown_callback():
        """Cleanup on shutdown - exit immediately, daemon threads handle rest"""
        logger.info("Agent session ending - uploads continue in background")
        logger.info("Agent session ended - process will exit now")

    ctx.add_shutdown_callback(shutdown_callback)
    await ctx.connect()

    agent = Agent(
        instructions=get_instruction(
            "Remote.ai",
            "Best AI company",
            "be the best ai company",
            "AI engineer",
            "Get best AI engineer",
            "Not provided",
            [],
            "Candidate"
        ),
        tools=[lookup_weather],
    )

    # Create session with real-time transcription support
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
                logger.info(f"Upload already in progress for {participant_identity}, skipping...")
                return

        participant_upload_in_progress[participant_identity] = True

        try:
            logger.info(f"Starting upload and cleanup workflow for {participant_identity}")

            # Step 1: Upload combined video to Azure
            logger.info(f"Uploading combined video to Azure...")
            video_upload_result = upload_video_to_azure(combined_video_path)

            if not video_upload_result.get('success'):
                logger.error(f"Failed to upload video: {video_upload_result.get('error')}")
                return

            video_url = video_upload_result['blob_url']
            logger.info(f"Video uploaded successfully: {video_url}")

            # Step 2: Upload transcript to Azure
            logger.info(f"Uploading transcript to Azure...")
            transcript_upload_result = upload_transcript_to_azure(transcript_path)

            if not transcript_upload_result.get('success'):
                logger.error(f"Failed to upload transcript: {transcript_upload_result.get('error')}")
                return

            transcript_url = transcript_upload_result['blob_url']
            logger.info(f"Transcript uploaded successfully: {transcript_url}")

            # Step 3: Save Azure links to file
            logger.info(f"Saving Azure links...")
            links_filename = None
            try:
                links_result = save_azure_links(participant_identity, video_url, transcript_url, auto_evaluate=False)
                if links_result and isinstance(links_result, dict) and links_result.get('links_file'):
                    links_filename = links_result.get('links_file')
                    logger.info(f"Links saved successfully: {links_filename}")
                else:
                    logger.warning(f"Links result format unexpected: {links_result}")
                    # Fallback: construct filename manually
                    links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
                    logger.info(f"Using fallback links filename: {links_filename}")
            except Exception as e:
                logger.error(f"Error saving links: {e}")
                import traceback
                traceback.print_exc()
                # Still try to construct filename for evaluation
                links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
                logger.info(f"Using fallback links filename after error: {links_filename}")

            # Step 3.5: Evaluation happens in disconnect handler, skip duplicate evaluation here
            logger.info(f"Evaluation will run automatically in disconnect handler")

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
                logger.info(f"All local files cleaned up successfully!")
            else:
                logger.warning(f"Some files could not be deleted")

            logger.info(f"Upload and cleanup workflow completed for {participant_identity}")

        except Exception as e:
            logger.error(f"Error in upload and cleanup workflow: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset upload flag
            participant_upload_in_progress[participant_identity] = False

    async def combine_video_audio(participant_identity: str):
        """Combine video and audio files into a single MP4 with perfect sync"""
        if participant_identity not in participant_recording_files:
            logger.warning(f"No recording files found for {participant_identity}")
            return

        files = participant_recording_files[participant_identity]
        video_file = files.get('video')
        audio_file = files.get('audio')

        if not video_file or not audio_file:
            logger.warning(f"Missing video or audio file for {participant_identity}")
            logger.info(f"Video: {video_file}")
            logger.info(f"Audio: {audio_file}")
            return

        # Check if both files exist
        if not os.path.exists(video_file) or not os.path.exists(audio_file):
            logger.error(f"Files not found:")
            logger.error(f"Video: {os.path.exists(video_file)} - {video_file}")
            logger.error(f"Audio: {os.path.exists(audio_file)} - {audio_file}")
            return

        # Create combined filename
        os.makedirs("recordings/combined", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = f"recordings/combined/{participant_identity}_{timestamp}_COMBINED.mp4"

        logger.info(f"Combining video and audio...")
        logger.info(f"Video: {video_file}")
        logger.info(f"Audio: {audio_file}")
        logger.info(f"Output: {combined_file}")

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
                logger.info(f"Successfully combined video and audio!")
                logger.info(f"Combined file: {combined_file}")

                # Check file size
                size_mb = os.path.getsize(combined_file) / (1024 * 1024)
                logger.info(f"File size: {size_mb:.2f} MB")

                # Track combined file for this participant
                if participant_identity not in participant_recording_files:
                    participant_recording_files[participant_identity] = {}
                participant_recording_files[participant_identity]['combined'] = combined_file
                logger.info(f"Video tracked and ready for upload (will be uploaded by disconnect handler)")
            else:
                logger.error(f"ffmpeg error:")
                logger.error(f"{result.stderr}")

        except FileNotFoundError:
            logger.error(f"ffmpeg not found! Install with: brew install ffmpeg")
        except Exception as e:
            logger.error(f"Error combining files: {e}")
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
            logger.info(f"Starting VIDEO recording for {participant_identity}")
            logger.info(f"Mode: TIMESTAMP TRACKING (for perfect audio sync)")

            # STEP 1: Collect ALL frames with timestamps
            async for event in video_stream:
                try:
                    video_frame: rtc.VideoFrame = event.frame
                    frame_count += 1
                    current_time = datetime.now()

                    # Use SHARED start time for this participant
                    if participant_identity not in participant_recording_times:
                        participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
                        logger.info(f"Recording START time set: {current_time}")

                    if start_time is None:
                        start_time = participant_recording_times[participant_identity]['start']

                    # Convert frame for recording (RGBA → BGR)
                    frame_rgba = video_frame.convert(rtc.VideoBufferType.RGBA)
                    frame_width = frame_rgba.width
                    frame_height = frame_rgba.height

                    # Skip frames with invalid dimensions (corrupted frames)
                    if frame_width < 100 or frame_height < 100:
                        logger.debug(f"Skipping corrupted frame {frame_count}: {frame_width}x{frame_height}")
                        continue

                    # Capture width/height from FIRST valid frame only
                    if width is None:
                        width = frame_width
                        height = frame_height
                        logger.info(f"Detected resolution: {width}x{height}")

                    # Resize frames to match target resolution instead of skipping
                    buffer = frame_rgba.data
                    img_rgba = np.frombuffer(buffer, dtype=np.uint8).reshape((frame_height, frame_width, 4))

                    # Resize if resolution changed
                    if frame_width != width or frame_height != height:
                        logger.debug(
                            f"Resizing frame {frame_count} from {frame_width}x{frame_height} to {width}x{height}")
                        img_rgba = cv2.resize(img_rgba, (width, height), interpolation=cv2.INTER_LINEAR)

                    img_array = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

                    # Store frame with its timestamp
                    frames_with_timestamps.append((current_time, img_array.copy()))
                    valid_frame_count += 1

                    # Log progress every 30 frames
                    if valid_frame_count % 30 == 0:
                        elapsed = (current_time - start_time).total_seconds()
                        logger.info(f"Recorded {valid_frame_count} valid frames ({elapsed:.1f}s)")

                except Exception as frame_error:
                    logger.error(f"Error processing frame {frame_count}: {frame_error}")
                    continue

        except asyncio.CancelledError:
            logger.info(f"Video recording cancelled (participant disconnected)")
            # Use SHARED end time set by disconnect handler
        except Exception as e:
            logger.error(f"Video stream error: {e}")
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
                logger.info(f"Video recording END time set: {end_time}")

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

                logger.info(f"Frame collection complete!")
                logger.info(f"Valid frames collected: {len(frames_with_timestamps)}")
                logger.info(f"Total recording duration: {total_duration:.1f} seconds")
                logger.info(f"Video resolution: {width}x{height}")

                # Validate resolution
                if width < 100 or height < 100:
                    logger.error(f"Invalid resolution: {width}x{height}")
                    return

                # STEP 3: Generate frames at regular intervals (30 FPS) to match audio duration
                target_frame_count = int(total_duration * target_fps)
                logger.info(f"Target frames for {total_duration:.1f}s @ {target_fps} FPS: {target_frame_count}")

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

                logger.info(f"Generated {len(output_frames)} frames (filled gaps for continuous playback)")

                # STEP 4: Write ALL frames to video
                os.makedirs("recordings/video", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"recordings/video/{participant_identity}_{timestamp}_VIDEO.mp4"

                # Use H.264 codec (avc1) for browser compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(video_filename, fourcc, target_fps, (width, height))

                if not video_writer.isOpened():
                    logger.error(f"Failed to open video writer!")
                    return

                logger.info(f"Writing {len(output_frames)} frames to: {video_filename}")
                logger.info(f"Settings: {width}x{height} @ {target_fps} FPS")

                # Write ALL frames
                for i, frame in enumerate(output_frames):
                    video_writer.write(frame)
                    if (i + 1) % 100 == 0 or (i + 1) == len(output_frames):
                        logger.info(f"Written {i + 1}/{len(output_frames)} frames...")

                video_writer.release()
                logger.info(f"Video writer released")

                logger.info(f"Video saved successfully!")
                logger.info(f"File: {video_filename}")
                logger.info(f"Actual frames captured: {len(frames_with_timestamps)}")
                logger.info(f"Output frames (with gap filling): {len(output_frames)}")
                logger.info(f"Video duration: {total_duration:.1f} seconds")
                logger.info(f"FPS: {target_fps}")
                logger.info(f"Gap filling: {len(output_frames) - len(frames_with_timestamps)} duplicate frames")
                logger.info(f"Video duration will MATCH audio duration perfectly!")

                # Track video filename for combining later
                if participant_identity not in participant_recording_files:
                    participant_recording_files[participant_identity] = {}
                participant_recording_files[participant_identity]['video'] = video_filename
                logger.info(f"Video filename tracked for combining")

                # Check if audio is also done - if yes, combine them!
                if 'audio' in participant_recording_files.get(participant_identity, {}):
                    logger.info(f"Both video and audio ready - combining now!")
                    task = asyncio.create_task(combine_video_audio(participant_identity))
                    background_tasks.append(task)
            else:
                logger.warning(f"No valid frames were recorded!")

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
            logger.info(f"Starting AUDIO recording for {participant_identity}")

            async for event in audio_stream:
                current_time = datetime.now()
                audio_frame: rtc.AudioFrame = event.frame
                audio_frame_count += 1

                # Use SHARED start time for this participant
                if participant_identity not in participant_recording_times:
                    participant_recording_times[participant_identity] = {'start': current_time, 'end': None}
                    logger.info(f"Recording START time set: {current_time}")

                if start_time is None:
                    start_time = participant_recording_times[participant_identity]['start']

                # Get audio properties from first frame
                if sample_rate is None:
                    sample_rate = audio_frame.sample_rate
                    num_channels = audio_frame.num_channels

                    os.makedirs("recordings/audio", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"recordings/audio/{participant_identity}_{timestamp}_AUDIO.wav"

                    logger.info(f"Recording to: {audio_filename}")
                    logger.info(f"Sample rate: {sample_rate}Hz, Channels: {num_channels}")

                # Store audio data (no processing, keep quality)
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
                audio_frames.append(audio_data.tobytes())

                # Log progress
                if audio_frame_count % 100 == 0:
                    logger.info(f"Audio: {audio_frame_count} frames")

        except asyncio.CancelledError:
            logger.info(f"Audio recording cancelled (participant disconnected)")
            # Use SHARED end time set by disconnect handler
        except Exception as e:
            logger.error(f"Audio error: {e}")
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
                logger.info(f"Audio recording END time set: {end_time}")

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
                            logger.info(f"Audio saved: {audio_filename} ({audio_frame_count} frames)")
                            logger.info(f"Audio uses SHARED timeline: {duration:.1f} seconds")

                    # Track audio filename for combining later
                    if participant_identity not in participant_recording_files:
                        participant_recording_files[participant_identity] = {}
                    participant_recording_files[participant_identity]['audio'] = audio_filename
                    logger.info(f"Audio filename tracked for combining")

                    # Check if video is also done - if yes, combine them!
                    if 'video' in participant_recording_files.get(participant_identity, {}):
                        logger.info(f"Both video and audio ready - combining now!")
                        task = asyncio.create_task(combine_video_audio(participant_identity))
                        background_tasks.append(task)

                except Exception as e:
                    logger.error(f"Error saving audio: {e}")
                    import traceback
                    traceback.print_exc()

    # Participant disconnect handler - ensures both video and audio stop together
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """Handle participant disconnect - IMMEDIATELY stop both video and audio"""
        disconnect_time = datetime.now()
        logger.info(f"Participant disconnected: {participant.identity}")




        # Set SHARED end time IMMEDIATELY
        if participant.identity in participant_recording_times:
            participant_recording_times[participant.identity]['end'] = disconnect_time
            logger.info(f"SHARED END time locked: {disconnect_time}")

            # Calculate final duration
            start = participant_recording_times[participant.identity]['start']
            duration = (disconnect_time - start).total_seconds()
            logger.info(f"Final recording duration: {duration:.1f} seconds")
            logger.info(f"Both video and audio will use this EXACT timeline")

        # Cancel recording tasks to stop immediately (not wait for streams)
        if participant.identity in participant_recording_tasks:
            tasks = participant_recording_tasks[participant.identity]
            for task_name, task in tasks.items():
                if task and not task.done():
                    task.cancel()
                    logger.info(f"Cancelled {task_name} recording task")
            logger.info(f"Both recordings stopped at SAME TIME: {disconnect_time}")


        # logger.info(f"Saving transcript for {participant.identity}...")

        # Save transcript, upload to Azure, and save links - then evaluation runs from Azure
        async def save_and_upload_workflow():
            """Save transcript, upload to Azure, save links, then evaluate from Azure"""

            # Start evaluation in background - don't wait for it
            async def run_evaluation_background():
                """Run evaluation in background and log when done"""
                try:
                    logger.info(f"Starting interview evaluation in background...")
                    evaluat = await evaluate_interview("/Users/danielsamuel/Documents/can_show_in_frontend/transcript_20251209_135356.json")
                    logger.info(f"Interview evaluation completed: {evaluat}")
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")
                    import traceback
                    traceback.print_exc()

            # Start evaluation as background task - continues immediately
            eval_task = asyncio.create_task(run_evaluation_background())
            background_tasks.append(eval_task)
            logger.info(f"Evaluation started in background - workflow continues...")

            try:
                logger.info(f"Saving transcript for {participant.identity}...")
                transcript_path = await write_transcript(session)






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
                                logger.info(f"Video ready: {combined_video_path}")
                                break
                        await asyncio.sleep(1)
                        waited += 1

                    # Upload to Azure in background thread (daemon - won't block exit)
                    def upload_and_save_links_background():
                        """Upload to Azure, save links JSON, trigger API evaluation"""
                        try:
                            import time
                            time.sleep(0.5)  # Brief delay

                            from azure_storage import upload_video_to_azure, upload_transcript_to_azure
                            from save_links import save_azure_links

                            # Upload transcript to Azure
                            logger.info(f"Uploading transcript to Azure...")
                            transcript_result = upload_transcript_to_azure(transcript_path)

                            if not transcript_result.get('success'):
                                logger.error(f"Failed to upload transcript: {transcript_result.get('error')}")
                                return

                            transcript_url = transcript_result['blob_url']
                            logger.info(f"Transcript uploaded: {transcript_url[:80]}...")

                            # Upload video if ready
                            video_url = None
                            if combined_video_path and os.path.exists(combined_video_path):
                                logger.info(f"Uploading video to Azure...")
                                video_result = upload_video_to_azure(combined_video_path)

                                if video_result.get('success'):
                                    video_url = video_result['blob_url']
                                    logger.info(f"Video uploaded: {video_url[:80]}...")
                                else:
                                    logger.warning(f"Video upload failed: {video_result.get('error')}")
                                    video_url = "pending"
                            else:
                                logger.warning(f"Video not ready yet, using placeholder")
                                video_url = "pending"

                            # Save links JSON (fast, no evaluation here)
                            logger.info(f"Saving Azure links to JSON file...")
                            result = save_azure_links(
                                participant_identity=participant.identity,
                                video_url=video_url,
                                transcript_url=transcript_url,
                                auto_evaluate=False  # DON'T evaluate in agent - too slow
                            )

                            if result and result.get('success'):
                                links_file = result.get('links_file')
                                logger.info(f"Links saved: {links_file}")

                                # Trigger evaluation via API (fire and forget)
                                try:
                                    import requests
                                    api_url = os.getenv('API_SERVER_URL', 'http://localhost:5001')
                                    logger.info(f"Triggering evaluation via API...")

                                    requests.post(
                                        f"{api_url}/api/evaluations/evaluate_from_links",
                                        json={
                                            "links_file": links_file,
                                            "participant_identity": participant.identity
                                        },
                                        timeout=2
                                    )
                                    logger.info(f"Evaluation API triggered")
                                except Exception as e:
                                    logger.warning(f"Failed to trigger evaluation API: {e}")
                            else:
                                logger.error(f"Failed to save links")

                        except Exception as e:
                            logger.error(f"Upload error: {e}")
                            import traceback
                            traceback.print_exc()

                    # Start upload in daemon thread (won't block exit, NOT tracked in background_tasks)
                    upload_thread = threading.Thread(target=upload_and_save_links_background, daemon=True,
                                                     name="UploadThread")
                    upload_thread.start()
                    logger.info(f"Upload started in background (independent daemon thread)")

                else:
                    logger.error(f"Failed to save transcript")
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()

        # Start transcript save, upload, and evaluation workflow
        task = asyncio.create_task(save_and_upload_workflow())
        background_tasks.append(task)

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
            logger.info(f"Audio track - starting independent recording")
            audio_task = asyncio.create_task(handle_audio_recording(track, participant.identity))
            participant_recording_tasks[participant.identity]['audio'] = audio_task

    logger.info("Starting agent session...")
    await session.start(agent=agent, room=ctx.room)

    # Generate initial greeting
    await session.generate_reply(instructions="greet the user and ask about their day")
    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


