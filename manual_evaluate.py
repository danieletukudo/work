#!/usr/bin/env python3
"""
Manual evaluation script for transcripts that weren't processed automatically.
Usage: python manual_evaluate.py <transcript_file_path>
"""

import sys
import os
from datetime import datetime
from prompt import load_transcript_from_file
from evaluate import evaluate_interview_comprehensive
from evaluation_formatter import save_evaluation_txt
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_evaluate.py <transcript_file_path>")
        print("Example: python manual_evaluate.py transcript_20251202_013528.json")
        sys.exit(1)
    
    transcript_path = sys.argv[1]
    
    if not os.path.exists(transcript_path):
        print(f" Transcript file not found: {transcript_path}")
        sys.exit(1)
    
    print(f" Loading transcript from: {transcript_path}")
    try:
        interview_transcript = load_transcript_from_file(transcript_path)
        print(f" Transcript loaded successfully")
    except Exception as e:
        print(f" Failed to load transcript: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract participant identity from filename or use default
    filename = os.path.basename(transcript_path)
    # Try to extract from transcript timestamp or use default
    participant_identity = "voice_assistant_user_be64e0f3"  # From the video filename
    
    # Get candidate info
    candidate_info = {
        "candidate_id": participant_identity,
        "candidate_email": f"{participant_identity}@example.com",
        "candidate_name": participant_identity.replace("_", " ").title(),
        "job_id": os.getenv("JOB_ID", "default_job")
    }
    
    # Get job description from env or use defaults
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
    
    print(f"\n Running comprehensive evaluation...")
    print(f"   Job: {job_description.get('title', 'N/A')}")
    print(f"   Candidate: {candidate_info.get('candidate_name', 'N/A')}")
    
    try:
        evaluation = evaluate_interview_comprehensive(
            job_description=job_description,
            candidate_cv={},  # Empty CV for now
            candidate_info=candidate_info,
            interview_transcript=interview_transcript,
            evaluation_instruction=None,
            use_api=False
        )
        
        if not evaluation:
            print(f" Evaluation returned None")
            sys.exit(1)
        
        print(f" Evaluation completed successfully")
        
        # Save evaluation to downloads directory
        downloads_dir = "downloads"
        os.makedirs(downloads_dir, exist_ok=True)
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_filename = f"evaluation_{participant_identity}_{current_date}.json"
        evaluation_path = os.path.join(downloads_dir, evaluation_filename)
        
        print(f"\n Saving evaluation to: {evaluation_path}")
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        # Also save as TXT
        txt_filename = evaluation_filename.replace('.json', '.txt')
        txt_path = os.path.join(downloads_dir, txt_filename)
        if save_evaluation_txt(evaluation, txt_path):
            print(f" TXT version saved: {txt_path}")
        
        print(f"\n Evaluation completed and saved!")
        print(f" Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
        print(f" Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A')}")
        print(f" JSON: {evaluation_path}")
        print(f" TXT: {txt_path}")
        
    except Exception as e:
        print(f" Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

