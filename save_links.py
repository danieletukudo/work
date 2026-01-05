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


    








    # Save links file
    links_filename = f"azure_links/{participant_identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_links.json"
    with open(links_filename, 'w') as f:
        json.dump({"video_url": video_url, "transcript_url": transcript_url}, f)

    print(f" Links saved to: {links_filename}")
    
    # Automatically trigger evaluation if enabled
    if auto_evaluate:
        print(f"\n Auto-evaluation enabled. Starting evaluation process...")
        print(f"   Links file: {links_filename}")
        try:
            result = download_and_evaluate_from_links(links_filename)
            
            # Verify result
            if result and result.get('success'):
                eval_file = result.get('evaluation_file_absolute') or result.get('evaluation_file')
                print(f"\n EVALUATION COMPLETE AND SAVED!")
                print(f"   File saved to: {eval_file}")
                if result.get('evaluation'):
                    print(f"   Overall Score: {result['evaluation'].get('overall_score', 'N/A')}/10")
            else:
                error = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"\n EVALUATION FAILED: {error}")
            
            return result
        except Exception as e:
            print(f" Auto-evaluation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Auto-evaluation failed: {str(e)}",
                "links_file": links_filename
            }
    else:
        return {
            "success": True,
            "links_file": links_filename,
            "message": "Links saved. Call download_and_evaluate_from_links() to run evaluation."
        }


def evaluate_local_transcript(
    transcript_path: str,
    participant_identity: str,
    job_description=None,
    candidate_cv=None,
    candidate_info=None,
    evaluation_instruction=None,
    use_api=False
):
    """
    Evaluate a locally saved transcript file directly (no download needed).
    
    This function:
    1. Loads the transcript from local file
    2. Runs comprehensive evaluation
    3. Saves evaluation result to JSON and TXT
    4. Returns the evaluation result
    
    Args:
        transcript_path: Path to the local transcript JSON file
        participant_identity: Participant identity (for candidate info)
        job_description: Optional job description dict (if None, will try to load from config)
        candidate_cv: Optional candidate CV dict (if None, will try to load from config)
        candidate_info: Optional candidate info dict (if None, will use participant_identity)
        evaluation_instruction: Optional custom evaluation instruction
        use_api: If True, use external API endpoint for evaluation
    
    Returns:
        dict: Evaluation results with keys:
            - success: bool
            - evaluation: dict (evaluation results if successful)
            - evaluation_file: str (path to saved evaluation JSON)
            - evaluation_txt_file: str (path to saved evaluation TXT)
            - error: str (error message if failed)
    """
    from evaluate import evaluate_interview_comprehensive
    from prompt import load_transcript_from_file
    from evaluation_formatter import save_evaluation_txt
    
    try:
        # Step 1: Verify transcript file exists
        if not os.path.exists(transcript_path):
            return {
                "success": False,
                "error": f"Transcript file not found: {transcript_path}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 2: Load transcript
        print(f" Loading transcript from: {transcript_path}")
        try:
            interview_transcript = load_transcript_from_file(transcript_path)
            print(f" Transcript loaded successfully")
        except Exception as e:
            print(f" Failed to load transcript: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to load transcript: {str(e)}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 3: Load job description, CV, and candidate info if not provided
        if not job_description or not candidate_cv or not candidate_info:
            # Try to load from config files
            participant_config_file = f"config_{participant_identity}.json"
            config_file = None
            
            if os.path.exists(participant_config_file):
                config_file = participant_config_file
            elif os.path.exists("config.json"):
                config_file = "config.json"
            
            if config_file:
                print(f" Loading config from: {config_file}")
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
            candidate_info = {
                "candidate_id": participant_identity,
                "candidate_email": f"{participant_identity}@example.com",
                "candidate_name": participant_identity.replace("_", " ").title(),
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
        
        # Step 4: Run evaluation
        print(f"\n Running evaluation...")
        print(f"   Job: {job_description.get('title', 'N/A')}")
        print(f"   Candidate: {candidate_info.get('candidate_name', 'N/A')}")
        print(f"   Transcript loaded: {len(str(interview_transcript))} characters")
        
        evaluation = None
        try:
            print(f"   Calling evaluate_interview_comprehensive...")
            evaluation = evaluate_interview_comprehensive(
                job_description=job_description,
                candidate_cv=candidate_cv or {},
                candidate_info=candidate_info,
                interview_transcript=interview_transcript,
                evaluation_instruction=evaluation_instruction,
                use_api=use_api
            )
            
            if not evaluation:
                raise Exception("Evaluation returned None")
            
            print(f" Evaluation completed successfully")
            print(f"   Evaluation keys: {list(evaluation.keys()) if isinstance(evaluation, dict) else 'Not a dict'}")
        except Exception as e:
            print(f" Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 5: Save evaluation result to local storage
        if not evaluation:
            print(f" Cannot save: evaluation is None")
            return {
                "success": False,
                "error": "Evaluation is None, cannot save",
                "evaluation": None,
                "evaluation_file": None
            }
        
        downloads_dir = "downloads"
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Get absolute path for clarity
        downloads_abs_path = os.path.abspath(downloads_dir)
        
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant_id = candidate_info.get("candidate_id", participant_identity)
        
        # Create evaluation filename with participant ID and timestamp
        evaluation_filename = f"evaluation_{participant_id}_{current_date}.json"
        evaluation_path = os.path.join(downloads_dir, evaluation_filename)
        evaluation_abs_path = os.path.abspath(evaluation_path)
        
        print(f"\n Saving evaluation result to local storage...")
        print(f"   Directory: {downloads_abs_path}")
        print(f"   Filename: {evaluation_filename}")
        print(f"   Full path: {evaluation_abs_path}")
        
        try:
            # Save evaluation to local file
            print(f"   Writing evaluation data to file...")
            with open(evaluation_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            
            print(f"   File write completed, verifying...")
            
            # Verify file was saved
            if os.path.exists(evaluation_path):
                file_size = os.path.getsize(evaluation_path)
                print(f" Evaluation saved successfully to local storage!")
                print(f"   Location: {evaluation_abs_path}")
                print(f"   Size: {file_size:,} bytes")
                
                # Double-check by reading it back
                try:
                    with open(evaluation_path, 'r', encoding='utf-8') as f:
                        test_read = json.load(f)
                    print(f"    File verified - can be read back successfully")
                except Exception as read_error:
                    print(f"    Warning: File exists but cannot be read: {read_error}")
            else:
                print(f" Evaluation file was not created: {evaluation_path}")
                print(f"   Current directory: {os.getcwd()}")
                print(f"   Downloads dir exists: {os.path.exists(downloads_dir)}")
                return {
                    "success": False,
                    "error": "Evaluation file was not created",
                    "evaluation": evaluation,
                    "evaluation_file": None
                }
        except Exception as e:
            print(f" Failed to save evaluation file: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to save evaluation file: {str(e)}",
                "evaluation": evaluation,
                "evaluation_file": None
            }
        
        # Step 6: Also save as TXT format
        txt_filename = evaluation_filename.replace('.json', '.txt')
        txt_path = os.path.join(downloads_dir, txt_filename)
        txt_abs_path = os.path.abspath(txt_path)
        
        print(f"\n Saving evaluation as TXT format...")
        try:
            if save_evaluation_txt(evaluation, txt_path):
                print(f" TXT version saved: {txt_abs_path}")
            else:
                print(f" Failed to save TXT version")
        except Exception as e:
            print(f" Error saving TXT version: {e}")
        
        print(f"\n EVALUATION SUMMARY:")
        print(f"   Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
        print(f"   Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10")
        print(f"   Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A')}")
        print(f"\n Evaluation results saved to local storage:")
        print(f"   JSON: {evaluation_abs_path}")
        print(f"   TXT:  {txt_abs_path}")
        
        return {
            "success": True,
            "evaluation": evaluation,
            "evaluation_file": evaluation_path,
            "evaluation_file_absolute": evaluation_abs_path,
            "evaluation_txt_file": txt_path,
            "evaluation_txt_file_absolute": txt_abs_path,
            "transcript_file": transcript_path,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error during evaluation: {str(e)}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "evaluation": None,
            "evaluation_file": None
        }


def download_and_evaluate_from_links(
    links_file_path: str,
    job_description=None,
    candidate_cv=None,
    candidate_info=None,
    evaluation_instruction=None,
    use_api=False
):
    """
    Download transcript from saved links file, run evaluation, and return results.
    
    This function:
    1. Reads the links JSON file
    2. Downloads the transcript from Azure
    3. Runs comprehensive evaluation
    4. Saves evaluation result to JSON
    5. Returns the evaluation result
    
    Args:
        links_file_path: Path to the links JSON file (e.g., "azure_links/user_20250124_links.json")
        job_description: Optional job description dict (if None, will try to load from config)
        candidate_cv: Optional candidate CV dict (if None, will try to load from config)
        candidate_info: Optional candidate info dict (if None, will try to load from config)
        evaluation_instruction: Optional custom evaluation instruction
        use_api: If True, use external API endpoint for evaluation
    
    Returns:
        dict: Evaluation results with keys:
            - success: bool
            - evaluation: dict (evaluation results if successful)
            - evaluation_file: str (path to saved evaluation JSON)
            - error: str (error message if failed)
    """
    from typing import Optional, Dict
    from azure_storage import download_video_from_azure
    from evaluate import evaluate_interview_comprehensive
    from prompt import load_transcript_from_file
    
    try:
        # Step 1: Read links file
        print(f"📖 Reading links file: {links_file_path}")
        with open(links_file_path, 'r') as f:
            links_data = json.load(f)
        
        transcript_url = links_data.get('transcript_url')
        if not transcript_url:
            return {
                "success": False,
                "error": "Transcript URL not found in links file",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 2: Download transcript
        print(f" Downloading transcript from Azure...")
        print(f"   Transcript URL: {transcript_url}")
        download_result = download_video_from_azure(transcript_url, download_folder="downloads")
        
        print(f"   Download result: {download_result}")
        
        if not download_result or not download_result.get('success'):
            error_msg = download_result.get('error', 'Unknown download error') if download_result else 'Download function returned None'
            print(f" Download failed: {error_msg}")
            return {
                "success": False,
                "error": f"Failed to download transcript: {error_msg}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        transcript_path = download_result.get('local_file_path')
        if not transcript_path:
            print(f" Download succeeded but no file path returned")
            return {
                "success": False,
                "error": "Download succeeded but no file path returned",
                "evaluation": None,
                "evaluation_file": None
            }
        
        print(f" Transcript downloaded: {transcript_path}")
        
        # Verify file exists
        if not os.path.exists(transcript_path):
            print(f" Downloaded file does not exist: {transcript_path}")
            return {
                "success": False,
                "error": f"Downloaded file does not exist: {transcript_path}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 3: Load transcript
        print(f" Loading transcript from: {transcript_path}")
        try:
            interview_transcript = load_transcript_from_file(transcript_path)
            print(f" Transcript loaded successfully")
        except Exception as e:
            print(f" Failed to load transcript: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to load transcript: {str(e)}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 4: Load job description, CV, and candidate info if not provided
        participant_identity = None
        if not job_description or not candidate_cv or not candidate_info:
            # Try to load from config files
            # Extract participant identity from links file name
            # Format: {participant_identity}_{timestamp}_links.json
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
                print(f" Loading config from: {config_file}")
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
        
        # Step 5: Run evaluation
        print(f"\n Running evaluation...")
        print(f"   Job: {job_description.get('title', 'N/A')}")
        print(f"   Candidate: {candidate_info.get('candidate_name', 'N/A')}")
        print(f"   Transcript loaded: {len(str(interview_transcript))} characters")
        
        evaluation = None
        try:
            print(f"   Calling evaluate_interview_comprehensive...")
            evaluation = evaluate_interview_comprehensive(
                job_description=job_description,
                candidate_cv=candidate_cv or {},
                candidate_info=candidate_info,
                interview_transcript=interview_transcript,
                evaluation_instruction=evaluation_instruction,
                use_api=use_api
            )
            
            if not evaluation:
                raise Exception("Evaluation returned None")
            
            print(f" Evaluation completed successfully")
            print(f"   Evaluation keys: {list(evaluation.keys()) if isinstance(evaluation, dict) else 'Not a dict'}")
        except Exception as e:
            print(f" Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "evaluation": None,
                "evaluation_file": None
            }
        
        # Step 6: Save evaluation result to local storage
        if not evaluation:
            print(f" Cannot save: evaluation is None")
            return {
                "success": False,
                "error": "Evaluation is None, cannot save",
                "evaluation": None,
                "evaluation_file": None
            }
        
        downloads_dir = "downloads"
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Get absolute path for clarity
        downloads_abs_path = os.path.abspath(downloads_dir)
        
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant_id = candidate_info.get("candidate_id", "unknown")
        
        # Create evaluation filename with participant ID and timestamp
        evaluation_filename = f"evaluation_{participant_id}_{current_date}.json"
        evaluation_path = os.path.join(downloads_dir, evaluation_filename)
        evaluation_abs_path = os.path.abspath(evaluation_path)
        
        print(f"\n Saving evaluation result to local storage...")
        print(f"   Directory: {downloads_abs_path}")
        print(f"   Filename: {evaluation_filename}")
        print(f"   Full path: {evaluation_abs_path}")
        
        try:
            # Save evaluation to local file
            print(f"   Writing evaluation data to file...")
            with open(evaluation_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)
            
            print(f"   File write completed, verifying...")
            
            # Verify file was saved
            if os.path.exists(evaluation_path):
                file_size = os.path.getsize(evaluation_path)
                print(f" Evaluation saved successfully to local storage!")
                print(f"   Location: {evaluation_abs_path}")
                print(f"   Size: {file_size:,} bytes")
                
                # Double-check by reading it back
                try:
                    with open(evaluation_path, 'r', encoding='utf-8') as f:
                        test_read = json.load(f)
                    print(f"    File verified - can be read back successfully")
                except Exception as read_error:
                    print(f"    Warning: File exists but cannot be read: {read_error}")
            else:
                print(f" Evaluation file was not created: {evaluation_path}")
                print(f"   Current directory: {os.getcwd()}")
                print(f"   Downloads dir exists: {os.path.exists(downloads_dir)}")
                return {
                    "success": False,
                    "error": "Evaluation file was not created",
                    "evaluation": evaluation,
                    "evaluation_file": None
                }
        except Exception as e:
            print(f" Failed to save evaluation file: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Failed to save evaluation file: {str(e)}",
                "evaluation": evaluation,
                "evaluation_file": None
            }
        
        # Step 7: Also save as TXT format
        txt_filename = evaluation_filename.replace('.json', '.txt')
        txt_path = os.path.join(downloads_dir, txt_filename)
        txt_abs_path = os.path.abspath(txt_path)
        
        print(f"\n Saving evaluation as TXT format...")
        try:
            from evaluation_formatter import save_evaluation_txt
            if save_evaluation_txt(evaluation, txt_path):
                print(f" TXT version saved: {txt_abs_path}")
            else:
                print(f" Failed to save TXT version")
        except Exception as e:
            print(f" Error saving TXT version: {e}")
        
        print(f"\n EVALUATION SUMMARY:")
        print(f"   Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
        print(f"   Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10")
        print(f"   Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A')}")
        print(f"\n Evaluation results saved to local storage:")
        print(f"   JSON: {evaluation_abs_path}")
        print(f"   TXT:  {txt_abs_path}")
        
        return {
            "success": True,
            "evaluation": evaluation,
            "evaluation_file": evaluation_path,
            "evaluation_file_absolute": evaluation_abs_path,
            "evaluation_txt_file": txt_path,
            "evaluation_txt_file_absolute": txt_abs_path,
            "transcript_file": transcript_path,
            "error": None
        }
        
    except FileNotFoundError as e:
        error_msg = f"Links file not found: {links_file_path}"
        print(f" {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "evaluation": None,
            "evaluation_file": None
        }
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in links file: {e}"
        print(f" {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "evaluation": None,
            "evaluation_file": None
        }
    except Exception as e:
        error_msg = f"Error during evaluation: {str(e)}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "evaluation": None,
            "evaluation_file": None
        }

