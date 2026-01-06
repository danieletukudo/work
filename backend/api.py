from fastapi import (FastAPI,
                     HTTPException,
                     BackgroundTasks,
                     status,
                     Query,
                     UploadFile,
                     File,
                     Body,
                     Form)
from cv_agent.database import DatabaseManager
import torch
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from cv_agent.donwload_and_process_cv import get_job_details
from cv_agent.batch_process import process_all_batches
from face_verifier.loggerfile import setup_logging as face_verifier_setup_logging
from cv_agent.loggerfile import setup_logging as cv_agent_setup_logging
from cv_agent.cv_agent_single import CandidateAnalyzer
from cv_agent.azure_cv_service import save_to_azure_blob
from cv_agent.azure_cv_service_v2 import (generate_sas_token,
                                          clean_database_metadata,
                                          delete_job_blobs_from_azure)
from personality_agent.database import DatabaseManager as PersonalityDb
from personality_agent.personality_type_extract import build_target_personality
from personality_agent.match_personality import calculate_match_score
from job_posting_agent.job_description import JobDesscriptionGenerator
from fastapi.middleware.cors import CORSMiddleware
torch.classes.__path__ = []
from fastapi import APIRouter, UploadFile, File, Form
from cv_agent.batch_process_blob_cvs import process_all_batches_blob
import httpx
import asyncio
import os


router = APIRouter()

load_dotenv()

 
# create fast api service
app = FastAPI(max_request_size=300 * 1024 * 1024)
 
# # Load environment variables from .env file
# load_dotenv()
 
face_verifier_logger = face_verifier_setup_logging("face_verifier_api")
cv_agent_logger = cv_agent_setup_logging("cv_agent_api")
cv_webhook_logger = cv_agent_setup_logging("cv_webhook_api")
single_cv_logger = cv_agent_setup_logging("single_cv_api")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400
)


# Add this after the existing models
class TopCandidatesRequest(BaseModel):
    limit: int = 10


# Upload directory for manual cv analysis
UPLOAD_FOLDER = "cv_agent/downloaded_cvs"


# In a utility file or at the top of your current file


WEBHOOK_URL = f"{os.getenv(CORE_APP_URL)}/webhook/job-application/cv-score"

async def send_webhook_score(job_id: str, candidate_id: str, fitness_score: float):
    """Sends the analysis score to the external webhook endpoint."""
    payload = {
        'job_application_id': job_id,
        'cv_score': fitness_score,
        'candidate_id': candidate_id,
    }
    
    # Use httpx.AsyncClient for asynchronous requests
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(WEBHOOK_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                cv_webhook_logger.info(f"Webhook success for candidate {candidate_id}. Status: {response.status_code}")
            else:
                cv_agent_logger.error(f"Webhook failed for candidate {candidate_id}. Status: {response.status_code}, Response: {response.text}")
                
    except httpx.RequestError as e:
        cv_webhook_logger.error(f"Webhook request failed for candidate {candidate_id}: {e}", exc_info=True)

 
@app.get("/")
def read_root():
    return {"message": "Welcome to Pegasi ai APIs"}

 
@app.get("/job-details/{job_id}")
def get_job(job_id: int):
    job_info = get_job_details(job_id)
    if not job_info:
        raise HTTPException(status_code=404,
                            detail="Job not found or no candidate data.")
    return job_info
 
@app.get("/job-details_manual_and_auto/{job_id}")
def get_job(job_id: int):
    db = DatabaseManager()
    try:
        cv_details = db.get_candidate_files(str(job_id))
    except Exception as e:
        print(f"Error retrieving candidate files: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    job_info = get_job_details(job_id)
    if not job_info:
        raise HTTPException(status_code=404,
                            detail="Job not found or no candidate data.")
    total_uploaded = len(cv_details)
    print(f"total uploaded cvs: {total_uploaded}")
    
    return {
        "job_info": job_info,
        "total_uploaded": total_uploaded
    }

@app.get("/job_status/{job_id}")
def job_status(job_id: int):
    db = DatabaseManager()
    try:
        current_status = db.get_job_status(str(job_id))
        return current_status
    except Exception as e:
        print(f"Error retrieving job status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/analyze-cvs/")
async def analyze(
    background_tasks: BackgroundTasks,
    job_id: Any = Form(...),
    instruction: str = Form(""),
    ignore_auto: str = Form(""),
    ):
    db = DatabaseManager()
    # Get job details from DB
    job_details = get_job_details(job_id)
    if not job_details:
        raise HTTPException(status_code=404, detail="Job not found")
    

    # Retrieve remote candidate details from job_details
    auto_candidates = job_details.get('candidates_details', [])
    total_auto = int(job_details.get('total_candidates', 0))
    manual_cv_details = db.get_candidate_files(str(job_id))

    print(f"manual cv details: {manual_cv_details[:4]}")
    
    
    # Retrieve manual candidate details if files have been uploaded
    manual_cv_urls = []
    if manual_cv_details:
        manual_cv_urls = [cv['url'] for cv in manual_cv_details]
        print(f"total manual cvs uploaded: {len(manual_cv_urls)}")
    
    manual_candidates= {}

    for cv_detail in manual_cv_details:
        candidate_id = cv_detail.get('blob_name').split('/')[-1].split('.')[0]
        manual_candidates[candidate_id] = {
            'url': cv_detail.get('url'),
            'email': '',
        }

    # Combine candidates: if both auto and manual exist, merge them
    total_candidates = len(auto_candidates) + len(manual_cv_urls)
    print(total_candidates)
    if total_candidates == 0:
        raise HTTPException(status_code=400, detail="No candidates found")
    

    # Update job status using one of the update paths (the total candidates remain the same)
    db.update_job_status(str(job_id), total_candidates, 0, "started", source="auto")
    
    # Launch background tasks based on candidate availability
    if ignore_auto == "False":
        print("analysing both manual and auto candidates")
        cv_agent_logger.info("Analyzing both manual and auto candidates")
        if auto_candidates and manual_cv_details:
            # If both available, process them separately
            background_tasks.add_task(
                process_all_batches,
                job_id,
                auto_candidates,
                job_details["job_description"],
                instruction,
                total_candidates,
                db
            )
            background_tasks.add_task(
                process_all_batches_blob,
                job_id,
                manual_candidates,
                job_details["job_description"],
                instruction,
                total_candidates,
                db
            )
        elif auto_candidates:
            background_tasks.add_task(
                process_all_batches,
                job_id,
                auto_candidates,
                job_details["job_description"],
                instruction,
                total_candidates,
                db
            )
            
            background_tasks.add_task(
                process_all_batches_blob,
                job_id,
                manual_candidates,
                job_details["job_description"],
                instruction,
                total_candidates,
                db
            )
     
    else:
        print("analyzing only manual candidates")
        cv_agent_logger.info("Analyzing only manual candidates")
        background_tasks.add_task(
                process_all_batches_blob,
                job_id,
                manual_candidates,
                job_details["job_description"],
                instruction,
                total_candidates,
                db
            )

    return {
        "message": "Processing started",
        "job_id": job_id,
        "total_cvs": total_candidates
    }


# endpoint to handle cv anlysis for individual candidates that applies to a job
# from the system
class singleCVDetails(BaseModel):
    job_description: object = Field(..., description="Job description object")
    candidate_id: str = Field(..., description="Candidate ID")
    candidate_email: str = Field(..., description="Candidate email")
    candidate_name: str = Field(..., description="Candidate name")
    job_id: int = Field(..., description="Job ID")
    candidate_cv: object = Field(
        None, description="json object with candidate cv details"
    )
    instruction: Optional[str] = Field("", description="Additional instructions")

# --- Pydantic Models for Final Output ---
class FinalAnalysisResult(BaseModel):
    """Pydantic model for the final analysis result to be saved."""
    fitness_score: float
    analysis_summary: Optional[str]
    candidate_id: str
    name: str
    platform_email: str
    email: str

@app.post("/analyze-single-cv/")
async def analyze_single_cv(
    background_tasks: BackgroundTasks,
    details: singleCVDetails = Body(...)
):
    job_id = int(details.job_id)
    instruction = details.instruction
    candidate_cv = details.candidate_cv
    candidate_id = details.candidate_id
    job_description = details.job_description

    # Get job details from DB
    analyzer = CandidateAnalyzer()

    output = await analyzer.analyze_cv(instruction, candidate_cv, job_description)
    result = output[0]
    print(f"Result for candidate {candidate_id}: {result}")
    db = DatabaseManager()
    
    analysis_result = FinalAnalysisResult(
        fitness_score=result.fitness_score,
        analysis_summary=result.analysis_summary,
        candidate_id=candidate_id,
        name=details.candidate_name,
        platform_email=details.candidate_email,
        email=details.candidate_email,
    )
    print(f"result is: {(analysis_result.name)}")

    db.save_candidate(
        str(job_id),
        analysis_result
    )
    # 3. Trigger Webhook in the background
    # Use asyncio.create_task to run the async webhook call non-blocking
    asyncio.create_task(
        send_webhook_score(
            job_id=job_id,
            candidate_id=candidate_id,
            fitness_score=result.fitness_score
        )
    )
    return {
        "message": f"Processing cv for candidate {candidate_id}",
        "job_id": job_id,
    }


@app.get("/candidate-result")
async def candidate_result(
    candidate_id:Any = Query(..., description = "the candidate is"),
    job_id:Any = Query(..., description = "the job id for the job posting analysed")):
    
    db = DatabaseManager()
    try:
        result = db.get_candidate_result(job_id, candidate_id)
        if not result:
            raise HTTPException(status_code=404, detail="Job ID or candidate id not found")
        return result
    finally:
        db.close


# Add new endpoints for retrieving results
@app.get("/analysis-status/{job_id}")
async def get_analysis_status(job_id: str):
    print(job_id)
    db = DatabaseManager()
    try:
        status = db.get_job_status(str(job_id))
        if not status:
            raise HTTPException(status_code=404, detail="Job ID not found")
        return status
    finally:
        db.close()
 
 
@app.get("/analysis-cost/{job_id}")
async def get_analysis_cost(job_id: int):
    db = DatabaseManager()
    try:
        cost = db.get_analysis_cost(str(job_id))
        details = {"job_id": job_id,
                   "cost": f"{(round(cost["cost"], 5)):.5f} USD",
                   "date": cost["created_at"]}
        if not cost:
            raise HTTPException(status_code=404, detail="Job ID not found")
        return details
    finally:
        db.close()
 
 
@app.get("/top-candidates/{job_id}")
async def get_top_candidates(
    job_id: str,
    top: Optional[int] = Query(default=None, ge=1, le=1000,
                               description="""Number of top
                               candidates to return""")
):
    db = DatabaseManager()
    try:
        candidates = db.get_top_candidates(str(job_id), top)
        return {
            "job_id": job_id,
            "candidates": candidates,
            "total_candidates": len(candidates),
            "top_limit": top if top else "all"
        }
    finally:
        db.close()
 

@app.post("/jobs/{job_id}/upload-cvs")
async def upload_cvs(
    job_id: str,
    files: list[UploadFile] = File(...)
):
    db = DatabaseManager()

    try:
        # MUST AWAIT async function
        uploaded_files = await save_to_azure_blob(files, job_id, db)
    except Exception as e:
        print(f"Error uploading files to Azure Blob: {e}")
        cv_agent_logger.error(f"Error uploading files to Azure Blob: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files to Azure Blob. {e}")

    try:
        # Save uploaded metadata
        await db.save_file_metadata(job_id, uploaded_files)
    except Exception as e:
        print(f"Error saving file metadata to database: {e}")
        cv_agent_logger.error(f"Error saving file metadata to database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file metadata. {e}")

    # Update job
    total_uploaded = len(uploaded_files)
    db.update_job_status(job_id, total_uploaded, 0, "ready_for_analysis", "manual")
    cv_agent_logger.info(f"Uploaded {total_uploaded} files for job {job_id}")

    return {
        "job_id": job_id,
        "message": "CVs uploaded successfully",
        "uploaded": len(uploaded_files),
        "status": "ready_for_analysis"
    }

# endpoint to get all existing file hashes for a job id
@app.get("/jobs/{job_id}/existing-file-hashes")
async def get_existing_file_hashes(job_id: str):
    db = DatabaseManager()
    try:
        existing_hashes = await db.get_existing_file_hashes(job_id)
        return {
            "job_id": job_id,
            "existing_file_hashes": existing_hashes
        }
    finally:
        db.close()


class FileUploadRecord(BaseModel):
    blob_name: str
    file_size: int
    file_hash: str

@app.get("/auth/upload-token/{job_id}")
async def get_upload_token(job_id: str):
    """
    Returns a SAS token so the frontend can upload directly.
    """
    try:
        # Check if job exists logic here...
        token_data = generate_sas_token(job_id)
        return token_data
    except Exception as e:
        cv_agent_logger.error(f"SAS generation error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate upload token")

@app.post("/jobs/{job_id}/register-uploads")
async def register_uploads(job_id: str, files: list[FileUploadRecord]):
    """
    Called by Frontend AFTER direct upload is complete to update DB.
    """
    db = DatabaseManager()
    
    # Format data for your existing DB structure
    metadata_list = []
    for f in files:
        # Note: We can't calculate MD5 here easily without downloading the file. 
        # For high-scale, you usually queue a background job to open the file and hash it/analyze it later.
        metadata_list.append({
            "blob_name": f.blob_name,
            "url": f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net/{os.getenv('BLOB_CONTAINER')}/{f.blob_name}",
            "file_hash": f.file_hash,
            "file_size": f.file_size,
        })

    await db.save_cv_metadata(job_id, metadata_list)
    
    # Update job status
    db.update_job_status(job_id, len(files), 0, "ready_for_analysis", "manual")
    
    return {"status": "synced", "count": len(files)}


@app.delete("/jobs/{job_id}/clear-cvs")
async def clear_job_data(job_id: str):
    """
    Deletes all uploaded CV files from Azure Blob Storage and clears 
    associated metadata from the backend system for a specific Job ID.
    
    This endpoint should be protected by appropriate authentication/authorization.
    """
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job ID must be provided."
        )

    # 1. Clean up database metadata (Blocking operation, runs synchronously)
    db_cleanup_success = clean_database_metadata(job_id)

    if not db_cleanup_success:
        # If metadata cleanup fails, you might want to stop here 
        # to prevent files being deleted without corresponding DB records.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear job metadata from the database."
        )

    # 2. Delete files from Azure Blob Storage (Runs asynchronously)
    try:
        deleted_count = await delete_job_blobs_from_azure(job_id)
    except Exception as e:
        # Log the error but still return success for metadata if files partially deleted
        print(f"Critical error during Azure deletion: {e}")
        deleted_count = 0 # Or track partial success
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata cleared, but file deletion failed. Error: {e}"
        )

    return {
        "job_id": job_id,
        "status": "success",
        "metadata_cleared": db_cleanup_success,
        "files_deleted_count": deleted_count,
        "message": f"Successfully deleted {deleted_count} CV files and cleared metadata for job {job_id}."
    }


@app.get("/failed-candidates/{job_id}")
async def get_failed_candidates(job_id: str):
    db = DatabaseManager()
    try:
        failed_candidates = db.get_failed_candidates(str(job_id))
        return {
            "job_id": job_id,
            "failed_candidates": failed_candidates,
            "total_failed": len(failed_candidates)
        }
    finally:
        db.close()
 

# endpoint to return all hr users
@app.get("/hr-users/")
async def get_hr_users():
    """
    Example of response 
    {"hr_users": [
        {"id": 1, "email": "abc@gmail.com", "name": "ABC", "created_at": "2024-10-01T12:00:00"},
        {"id": 2, "email": "adb@gmail.com, "name": "ADB", "created_at": "2024-10-02T12:00:00"}
    ],
    "total_users": 2
    }
    """
    db = DatabaseManager()
    try:
        hr_users = db.fetch_hr_users()
        return {
            "hr_users": hr_users,
            "total_users": len(hr_users)
        }
    finally:
        db.close()


 ## JOB POSTING AGENT ENDPOINTS ###

# --- Pydantic Model for Input Validation ---
class JobDetails(BaseModel):
    job_title: str
    skills: List[str]
    job_level: str
    industries: Optional[List[str]]
    job_functions: Optional[List[str]]
    work_type: Optional[str]
    timezone: Optional[str]
    other_details: str


# --- Job Description Endpoint---
@app.post("/generate-job-description")
async def generate_job_description(details: JobDetails):
    """
    Generates a job description based on employer input using Gemini.
    """
    try:
        # Construct the prompt for the Gemini model
        print("job description generation request received")
        prompt = f"""
         Input Details:
        - Job Title: {details.job_title}
        - Job Level: {details.job_level}
        - Work Type: {details.work_type if details.work_type else 'None'}
        - Time Zone: {details.timezone if details.work_type else 'None'}
        - Industries: {', '.join(details.industries) if details.industries else 'None'}
        - Job Functions: {', '.join(details.job_functions) if details.job_functions else 'None'}
        - Required Skills: {', '.join(details.skills)}
        - Additional Details: {details.other_details if details.other_details else 'None'}

        Output Format:
        The output should follow the format specified, Use clear headings and bullet points.
        Do not include any conversational text outside of the job description itself.
        """
        generator = JobDesscriptionGenerator()
        response, _, __ = await generator.generate_job_description(prompt)

        if response:
            return {"job_description": response}
        else:
            raise HTTPException(status_code=500, detail="Gemini did not return any text.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during content generation: {e}")


## PERSONALITY ENDPOINTS ##
class PersonalityPreferenceInput(BaseModel):
    job_posting_id: str
    traits: Dict[str, int]
    description: str

@app.get("/candidate_personality/{candidate_id}")
async def get_candidate_personality(candidate_id: str):
    db = PersonalityDb()
    try:
        personality = db.get_candidate_personality(candidate_id)
        if not personality:
            raise HTTPException(status_code=404, detail="Candidate personality not found.")
        return {
            "candidate_id": candidate_id,
            "personality_profile": personality
        }
    finally:
        db.close()

@app.get("/candidate-personality-exists/{candidate_id}")
async def personality_exists(candidate_id: str):
    db = PersonalityDb()
    try:
        personality = db.get_candidate_personality(candidate_id)
        if personality:
            return {"status": "user_exists"}
        else:
            return {"status": "no_user"}
    finally:
        db.close()


@app.post("/save-personality-preference")
async def save_personality_preference(preference: PersonalityPreferenceInput):
    db = PersonalityDb()
    try:
        success = db.save_personality_preferences(
            job_posting_id=preference.job_posting_id,
            traits=preference.traits,
            description=preference.description
        )
        if success:
            print("saved successfully")
            result = await build_target_personality(preference.job_posting_id)
            print(f"target personality: {result}")
            return {"message": "Personality preferences saved successfully."}
        else:
            print("failed to save")
            raise HTTPException(status_code=500, detail="Failed to save personality preferences.")
    finally:
        db.close()

class PersonalityData:
    # Define your question-to-trait mapping
    QUESTION_MAP = {
        "openness": list(range(1, 16)),
        "conscientiousness": list(range(16, 31)),
        "extraversion": list(range(31, 46)),
        "agreeableness": list(range(46, 61)),
        "neuroticism": list(range(61, 76)),
    }

    # Define reverse-scored questions
    REVERSE_SCORED = {
        6, 13, 14, 21, 26, 28, 30, 36, 41, 44, 51, 56, 59, 66, 71, 74
    }

class CandidatePersonalityResult(BaseModel):
    user_id: str
    responses: Dict[str, int] 

@app.post("/api/personality-score")
async def score_personality(data: CandidatePersonalityResult):
    db = PersonalityDb()
    responses = data.responses
    print(f"responses: {responses}")
    scores = {}

    # Calculate scores per trait
    for trait, ids in PersonalityData.QUESTION_MAP.items():
        trait_scores = []
        for qid in ids:
            if str(qid) in responses:
                val = responses[str(qid)]

            else:
                continue

            # handle reverse scoring
            if qid in PersonalityData.REVERSE_SCORED:
                val = 6 - val
            trait_scores.append(val)
            print(f"Question ID: {qid}, Original Value: {responses[str(qid)]}, Scored Value: {val}")

        if len(trait_scores) > 0:
            avg = sum(trait_scores) / len(trait_scores)
            percent = round(((avg - 1) / 4) * 100, 2)
        else:
            percent = 0.0
        scores[trait] = percent

    db.save_candidate_personality(data.user_id, scores)

    return {
        "success": True,
        "user_id": data.user_id,
        "scores": scores,
    }

class CandidateInput(BaseModel):
    candidate_id: str
    name: str
    personality: Dict[str, float]
    job_posting_id: str

@app.post("/webhook/match-personality/")
async def webhook_match_personality(candidate: CandidateInput):
    try:
        # Create candidate dict (exclude job_posting_id from candidate info)
        candidate_data = {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "personality": candidate.personality
        }
        # Calculate the match score using the provided candidate data and job posting id
        score_dict = await calculate_match_score(candidate_data, candidate.job_posting_id)
        print(f"calculated score: {score_dict}")
        return score_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating match score: {e}")