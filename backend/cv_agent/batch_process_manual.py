from fastapi import HTTPException
from cv_agent.database import DatabaseManager
import shutil
import asyncio
from pydantic import BaseModel
from cv_agent.cv_agent1 import CVProcessor, CVAnalyzerException
import math
from typing import Dict, List, Any
from cv_agent.donwload_and_process_cv import process_cv_folder as process_manual_uploaded_files
from cv_agent.loggerfile import setup_logging
import os

# setup logging
logger = setup_logging("batch_process_manual")

# models
class TopCandidatesRequest(BaseModel):
    limit: int = 1000

# CONSTANTS
BATCH_SIZE = 3
MAX_CONCURRENT_BATCHES = 4
DOWNLOAD_FOLDER_PREFIX = "manual_cvs_"

# Function to process a single manual batch
async def process_manual_cv_batch(
    candidate_details: Dict[str, Dict[str, Any]],
    job_description: str,
    instruction: str,
    batch_id: int,
    job_id: int,
    total_candidates_in_job: int,
    db: DatabaseManager
) -> None:
    """
    Processes a single batch of manually uploaded CVs.
    """

    try:

        # Create a subset of the candidate_details dictionary for this batch
        batch_candidate_details = candidate_details

        if not batch_candidate_details:
            logger.warning(f"No candidates provided for manual batch {batch_id}. Skipping analysis.")
            return
        logger.info(f"Processing manual batch {batch_id} for job {job_id}")
        # Process the manually uploaded CVs (no download step is needed)
        # Assuming `process_manual_uploaded_files` handles the file-to-text conversion
        result = process_manual_uploaded_files(
            batch_candidate_details,
            db,
            job_id
        )
        if isinstance(result, tuple) and len(result) == 2:
            processed_candidates, failed_candidates_processing = result
        else:
            processed_candidates = result
            failed_candidates_processing = []
        
        if not processed_candidates:
            logger.warning(f"No candidates processed for manual batch {batch_id}. Skipping analysis.")
            return

        # Get results using the processed CV texts
        logger.info(f"Starting CV analysis for manual batch {batch_id}")
        processor = CVProcessor()
        results, cost = await processor.process_candidates(
            processed_candidates,
            job_description,
            instruction,
            db,
            job_id
        )
        
        if results:
            db.save_analysis_cost(str(job_id), cost)
            logger.info(f"Saving analysis cost for manual batch {batch_id}: ${cost:.4f}")

            logger.info(f"Saving {len(results)} analysis results for manual batch {batch_id} to database.")
            for candidate in results:
                try:
                    db.save_candidate(str(job_id), candidate)
                    logger.debug(f"Candidate {candidate.candidate_id} saved successfully.")
                except Exception as e:
                    logger.error(f"Failed to save candidate {candidate.candidate_id} from batch {batch_id}: {e}", exc_info=True)
            
            # Update job status incrementally
            current_job_status = db.get_job_status(str(job_id))
            if current_job_status:
                processed_so_far = current_job_status.get('manual_processed_cvs', 0) + len(results)
                db.update_job_status(
                    job_id=str(job_id),
                    total_cvs=total_candidates_in_job,
                    processed_cvs=processed_so_far,
                    status="processing",
                    source="manual"
                )
                logger.debug(f"Job {job_id} status updated: {processed_so_far}/{total_candidates_in_job} manual-processed.")
        else:
            logger.warning(f"No successful analysis results returned for manual batch {batch_id} of job {job_id}.")

    except CVAnalyzerException as e:
        logger.error(f"CV analysis error in manual batch {batch_id} for job {job_id}: {str(e)}", exc_info=True)
        error_msg = str(e)
        # Check for the specific API quota error
        if "429 RESOURCE_EXHAUSTED" in error_msg:
            current_job_status = db.get_job_status(str(job_id)) if db.get_job_status(str(job_id)) else {}
            processed_so_far = current_job_status.get('manual_processed_cvs', 0) if current_job_status else 0
            logger.error(f"API quota exceeded for job {job_id}. Aborting analysis.")
            db.update_job_status(
                job_id=str(job_id),
                total_cvs=total_candidates_in_job,
                processed_cvs=processed_so_far,
                status="failed due to API quota exhaustion",
                source="manual"
            )
            # Raise an HTTPException with a 429 status code for the frontend
            raise HTTPException(status_code=429, detail="API quota exceeded. Please contact the dev team.")
        else:
            # Handle other CV analysis errors gracefully
            logger.error(f"CV analysis error in batch {batch_id} for job {job_id}: {error_msg}", exc_info=True)
            raise
        # Let other batches continue if this one fails
    except Exception as e:
        logger.error(f"Unexpected error processing manual batch {batch_id} for job {job_id}: {str(e)}", exc_info=True)
        # Let other batches continue


# Function to process all manual batches
async def manual_process_all_batches(
    job_id: int,
    candidate_details: Dict[str, Dict[str, Any]],
    job_description: str,
    instruction: str,
    total_candidates: int,
    db: DatabaseManager
):
    """
    Orchestrates the batch processing for all manually uploaded CVs.
    """
    logger.info(f"Starting analysis for manual job {job_id} with {total_candidates} CVs.")
    
    # Initialize job status
    db.update_job_status(str(job_id), total_candidates, 0, "processing", "manual")

    all_candidate_items = list(candidate_details.items())
    num_cvs = len(all_candidate_items)
    num_batches = math.ceil(num_cvs / BATCH_SIZE)

    try:
        for i in range(0, num_batches, MAX_CONCURRENT_BATCHES):
            batch_tasks = []
            for j in range(MAX_CONCURRENT_BATCHES):
                batch_id = i + j
                if batch_id < num_batches:
                    start_idx = batch_id * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, num_cvs)
                    
                    batch_candidate_items = all_candidate_items[start_idx:end_idx]
                    batch_candidate_details = {
                        cid: details for cid, details in batch_candidate_items
                    }

                    if batch_candidate_details:
                        task = process_manual_cv_batch(
                            candidate_details=batch_candidate_details,
                            job_description=job_description,
                            instruction=instruction,
                            batch_id=batch_id,
                            job_id=job_id,
                            total_candidates_in_job=total_candidates,
                            db=db
                        )
                        batch_tasks.append(task)
            
            if batch_tasks:
                logger.debug(f"Waiting for manual batch group {i//MAX_CONCURRENT_BATCHES + 1} to complete.")
                await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        logger.info(f"Manual job {job_id} completed successfully. Updating final status.")
        db.update_job_status(
            job_id=str(job_id),
            total_cvs=total_candidates,
            processed_cvs=num_cvs,
            status="completed",
            source="manual"
        )
        
    except Exception as e:
        logger.error(f"Manual job {job_id} failed: {str(e)}", exc_info=True)
        db.update_job_status(
            job_id=str(job_id),
            total_cvs=total_candidates,
            processed_cvs=db.get_job_status(str(job_id)).get('manual_processed_cvs', 0) if db.get_job_status(str(job_id)) else 0,
            status="failed",
            source="manual"
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Final cleanup in case of unexpected errors
        for i in range(num_batches):
            batch_folder_path = f"{DOWNLOAD_FOLDER_PREFIX}{job_id}_{i}"
            if os.path.exists(batch_folder_path):
                shutil.rmtree(batch_folder_path, ignore_errors=True)
