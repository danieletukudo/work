from fastapi import HTTPException
from cv_agent.database import DatabaseManager
import asyncio
from pydantic import BaseModel
from cv_agent.cv_agent1 import CVProcessor, CVAnalyzerException
import math
import shutil
import os # Import os for path manipulation
from cv_agent.donwload_and_process_cv import download_cvs, process_cv_folder, download_cvs_manual
from cv_agent.loggerfile import setup_logging
from typing import Dict

# setup logging
logger = setup_logging("batch_process_blob_cvs")

# --- Models ---
class TopCandidatesRequest(BaseModel):
    limit: int = 1000

# --- CONSTANTS ---
BATCH_SIZE = 20  # Number of CVs to process in each batch
MAX_CONCURRENT_BATCHES = 4  # Maximum number of batches to process simultaneously
DOWNLOAD_FOLDER_PREFIX = "downloaded_cvs_" # Prefix for temporary download folders

# --- Helper Functions (for better readability and reusability) ---
async def _process_single_batch(
    batch_id: int,
    job_id: int,
    candidate_details_batch: Dict, # Changed to accept a pre-filtered dict
    job_description: str,
    instruction: str,
    db: DatabaseManager,
    total_candidates_in_job: int, # Pass total for job status update
) -> None:
    """
    Processes a single batch of CVs: downloads, processes, analyzes, and saves results.
    """
    batch_folder_path = f"{DOWNLOAD_FOLDER_PREFIX}{job_id}_{batch_id}"
    
    try:
        logger.info(f"Processing batch {batch_id} for job {job_id}. Candidates: {list(candidate_details_batch.keys())}")

        # Download CVs for this batch
        logger.info(f"Downloading CVs to {batch_folder_path} for batch {batch_id}")
        downloaded_candidate_details, failed_links = download_cvs_manual(candidate_details_batch, batch_folder_path) # Pass the batch_folder_path
        
        if not downloaded_candidate_details:
            logger.warning(f"No CVs downloaded for batch {batch_id} for job {job_id}. Skipping processing.")
            return

        # Process the downloaded CVs
        logger.info(f"Processing downloaded CVs from {batch_folder_path} for batch {batch_id}")
        result = process_cv_folder(downloaded_candidate_details, db, job_id)
        if isinstance(result, tuple) and len(result) == 2:
            processed_candidates_data, failed_candidates_processing = result
        else:
            processed_candidates_data = result
            failed_candidates_processing = []

        if not processed_candidates_data:
            logger.warning(f"No candidates processed from folder {batch_folder_path} for batch {batch_id}. Skipping analysis.")
            return

        # Get results using the processed CV texts
        logger.info(f"Starting CV analysis for batch {batch_id} ({len(processed_candidates_data)} candidates)")
        processor = CVProcessor()
        results, cost = await processor.process_candidates(
            processed_candidates_data,
            job_description,
            instruction
        )
        
        if results:
            # Saving cost per batch as returned by processor.
            db.save_analysis_cost(str(job_id), cost)
            logger.info(f"Saving analysis cost for batch {batch_id}: ${cost:.4f}")

            # Save results to database
            logger.info(f"Saving {len(results)} analysis results for batch {batch_id} to database.")
            for candidate_result in results:
                try:
                    db.save_candidate(str(job_id), candidate_result)
                except Exception as e:
                    logger.error(f"Failed to save candidate {candidate_result.candidate_id} from batch {batch_id}: {e}", exc_info=True)
            
            # Update job status (incrementally)
            # This logic needs to be careful with concurrency to avoid race conditions
            # A simpler approach for `processed_cvs`: get the current job status, add count of current batch, then update.
            # Assuming `update_job_status` handles increments properly or you manage it atomically.
            current_job_status = db.get_job_status(str(job_id))
            if current_job_status:
                auto_processed_cvs_so_far = current_job_status.get('auto_processed_cvs', 0) + len(results)
                db.update_job_status(
                    job_id=str(job_id),
                    total_cvs=total_candidates_in_job,
                    processed_cvs=auto_processed_cvs_so_far,
                    status="processing", # Still processing until all batches are done
                    source="auto"
                )
                logger.debug(f"Job {job_id} status updated: {auto_processed_cvs_so_far}/{total_candidates_in_job} auto-processed.")
        else:
            logger.warning(f"No successful analysis results returned for batch {batch_id} of job {job_id}.")

    except CVAnalyzerException as e:
        logger.error(f"CV analysis error in batch {batch_id} for job {job_id}: {str(e)}", exc_info=True)
        error_msg = str(e)
        # Check for the specific API quota error
        if "429 RESOURCE_EXHAUSTED" in error_msg:
            logger.error(f"API quota exceeded for job {job_id}. Aborting analysis.")
            current_job_status = db.get_job_status(str(job_id)) if db.get_job_status(str(job_id)) else {}
            auto_processed_cvs_so_far = current_job_status.get('auto_processed_cvs', 0) if current_job_status else 0
            db.update_job_status(
                job_id=str(job_id),
                total_cvs=total_candidates_in_job,
                processed_cvs=auto_processed_cvs_so_far,
                status="failed due to API quota exhaustion",
                source="auto"
            )
            # Raise an HTTPException with a 429 status code for the frontend
            raise HTTPException(status_code=429, detail="API quota exceeded. Please contact the dev team.")
        else:
            # Handle other CV analysis errors gracefully
            logger.error(f"CV analysis error in batch {batch_id} for job {job_id}: {error_msg}", exc_info=True)
            raise
        # Don't re-raise immediately; let other batches continue if possible
    except Exception as e:
        logger.error(f"Unexpected error processing batch {batch_id} for job {job_id}: {str(e)}", exc_info=True)
        # Log the error but don't prevent other batches from potentially running
    finally:
        # Ensure cleanup happens even if errors occur
        if os.path.exists(batch_folder_path):
            logger.debug(f"Cleaning up temporary files in {batch_folder_path} for batch {batch_id}")
            shutil.rmtree(batch_folder_path, ignore_errors=True)
            logger.debug(f"Temporary files for batch {batch_id} cleaned up.")


async def process_all_batches_blob(
    job_id: int,
    candidate_details: Dict, # This should be the full dict of all candidates for the job
    job_description: str,
    instruction: str,
    total_candidates: int,
    db: DatabaseManager
):
    """
    Processes all CV batches for a job with controlled concurrency and database integration.
    """
    logger.info(f"Starting full analysis for job {job_id} with {total_candidates} CVs.")

    # Convert candidate_details dict to a list of (candidate_id, details) tuples for easier slicing
    all_candidate_items = list(candidate_details.items())
    num_cvs = len(all_candidate_items)
    num_batches = math.ceil(num_cvs / BATCH_SIZE)

    # Initialize job status if not already
    current_job_status = db.get_job_status(str(job_id))
    if not current_job_status or current_job_status.get('status') == 'pending':
        db.update_job_status(str(job_id), total_candidates, 0, "processing", "auto")
        logger.info(f"Initialized job {job_id} status to 'processing'.")
    else:
        logger.info(f"Job {job_id} already in status '{current_job_status.get('status')}', resuming processing.")


    try:
        # Use asyncio.Semaphore to limit concurrent calls to _process_single_batch
        # This controls the number of 'active' batch processes at any given time.
        # It's different from `MAX_CONCURRENT_BATCHES` but can complement it.
        # Here, `MAX_CONCURRENT_BATCHES` directly limits the `asyncio.gather` groups.
        
        # Process batches in groups to control overall concurrency
        for i in range(0, num_batches, MAX_CONCURRENT_BATCHES):
            batch_tasks = []
            current_batch_group_limit = min(MAX_CONCURRENT_BATCHES, num_batches - i)
            logger.debug(f"Starting batch group {i//MAX_CONCURRENT_BATCHES + 1} with {current_batch_group_limit} batches.")

            for j in range(current_batch_group_limit):
                batch_id = i + j
                start_idx = batch_id * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, num_cvs) # Ensure end_idx doesn't exceed num_cvs
                
                # Get the slice of candidate items for this specific batch
                batch_candidate_items = all_candidate_items[start_idx:end_idx]
                
                # Reconstruct the dictionary for the batch
                batch_candidate_details = {
                    cid: details for cid, details in batch_candidate_items
                }

                if not batch_candidate_details:
                    logger.warning(f"Batch {batch_id} is empty. Skipping task creation.")
                    continue

                task = _process_single_batch(
                    batch_id=batch_id,
                    job_id=job_id,
                    candidate_details_batch=batch_candidate_details,
                    job_description=job_description,
                    instruction=instruction,
                    db=db,
                    total_candidates_in_job=total_candidates,
                )
                batch_tasks.append(task)
            
            if batch_tasks:
                logger.debug(f"Waiting for batch group {i//MAX_CONCURRENT_BATCHES + 1} ({len(batch_tasks)} tasks) to complete.")
                # Wait for current batch group to complete. If any individual batch fails,
                # `_process_single_batch` logs it, but `gather` will still wait for all.
                # If you want `gather` to fail fast on any error, remove `try/except` from `_process_single_batch`.
                await asyncio.gather(*batch_tasks, return_exceptions=True) # return_exceptions allows other tasks to finish
                                                                           # and you can inspect results for errors

        logger.info(f"Job {job_id} completed all batches successfully. Updating final status.")
        db.update_job_status(str(job_id), total_candidates, 0, "complete", source="manual")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed during batch processing: {str(e)}", exc_info=True)
        # Update status to failed if something goes wrong in the overall process
        db.update_job_status(
            job_id=str(job_id),
            total_cvs=total_candidates,
            processed_cvs=db.get_job_status(str(job_id)).get('auto_processed_cvs', 0) if db.get_job_status(str(job_id)) else 0, # Report processed count so far
            status="failed",
            source="manual"
        )
        # Re-raise as HTTPException for API endpoint
        raise HTTPException(status_code=500, detail=f"Job processing failed: {str(e)}")
    finally:
        # Ensure any remaining temporary folders are cleaned up in case of unexpected termination
        # This loop isn't strictly necessary if _process_single_batch cleans up immediately
        # but provides a fallback for robustness.
        for i in range(num_batches):
            batch_folder_path = f"{DOWNLOAD_FOLDER_PREFIX}{job_id}_{i}"
            if os.path.exists(batch_folder_path):
                logger.warning(f"Found and cleaning up leftover temporary folder: {batch_folder_path}")
                shutil.rmtree(batch_folder_path, ignore_errors=True)
