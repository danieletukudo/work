import evaluate
import os
from prompt import load_transcript_from_file
import json
from evaluate import evaluate_interview_comprehensive
import asyncio


def evaluate_interview(
        links_file_path,
        job_description=None,
        candidate_cv=None,
        candidate_info=None,
        evaluation_instruction=None,
        use_api=False
):


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
    evaluation = evaluate_interview_comprehensive(
        job_description=job_description,
        candidate_cv=candidate_cv or {},
        candidate_info=candidate_info,
        interview_transcript=interview_transcript,
        evaluation_instruction=evaluation_instruction,
        use_api=use_api
    )

    # Return formatted strings and full evaluation dict for webhook
    return (
        f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10",
        f"Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10",
        evaluation  # Return full evaluation dict as third element for webhook
    )



print(evaluate_interview("transcript_20251219_144526.json"))