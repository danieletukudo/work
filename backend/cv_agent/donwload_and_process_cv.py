import requests
import os
from dotenv import load_dotenv
from .process_doc import (extract_text_from_pdf,
                          extract_text_from_docx,
                          extract_text_from_doc)
import glob
from cv_agent.loggerfile import setup_logging   
import shutil
from cv_agent.database import DatabaseManager
import re # Import the regular expression module
import uuid


# setup logging
logger = setup_logging("download_and_process_cv")

load_dotenv()

UPLOAD_FOLDER = "cv_agent/downloaded_cvs"
BASE_URL = os.getenv("BASE_URL")
SAS_TOKEN = os.getenv("SAS_TOKEN")
CORE_APP_URL = os.getenv("CORE_APP_URL")

headers = {
        "User-Agent": os.getenv("USER_AGENT"),
        "Referer": os.getenv("REFERER"),
        "Accept": os.getenv("ACCEPT"),
        "Accept-Language": os.getenv("ACCEPT_LANGUAGE"),
        "Connection": "keep-alive"
        }

# Function to get job details from the API
def get_job_details(job_id):
    # Define the URL using the provided job ID

    url = f"{CORE_APP_URL}/jobs/{job_id}/candidates/pending-candidate"
    logger.info(f"Fetching job details for job ID: {job_id}")
    logger.debug(f"API URL: {url}")

    try:
        # Make a GET request to the API
        response = requests.get(url, headers=headers, allow_redirects=True)
        # Check if the response is successful
        if response.status_code == 200:
            # Parse the JSON response  
            data = response.json()         
            # Extract job details
            job_title = data.get("job").get("title")
            job_description = data.get("job").get("description")
            employer_email = data.get('employer', {}).get('email')
            # total_candidates = data.get('candidates', {}).get('total', 0)
            
            # Extract candidate details (CV link)
            candidates = data.get('candidates', {})
            candidate_details = {}
            for candidate in candidates:
                candidate_id = candidate.get('candidate_id')
                candidate_cv = candidate.get('candidate_details', {}).get('resume')
                candidate_email = candidate.get('candidate_details', {}).get('email')
                candidate_details[candidate_id] = {
                    'cv': candidate_cv,
                    'email': candidate_email
                }
            
            total_candidates = len(candidate_details.keys())
            logger.info(f"Successfully fetched job details. Found {total_candidates} candidates")
            logger.debug(f"Job Title: {job_title}")
            # candidate_cv = [candidate['candidate_details'].get('resume') for candidate in candidates if 'resume' in candidate['candidate_details']]
            
            # Return the extracted details as a dictionary
            return {
                'job_title': job_title,
                'job_description': job_description,
                'employer_email': employer_email,
                'candidates_details': candidate_details,
                'total_candidates': len(candidate_details.keys())
            }
        else:
            # If the response is not successful, print the status code
            error_msg = f"Failed to fetch data.{response.json()['message']}. Status code: {response.status_code}"
            logger.error(error_msg)
            return response.json()['message']
    except Exception as e:
        logger.error(f"Error fetching job details: {str(e)}", exc_info=True)
        return None

# Function to clean up the filename for safer storage
def sanitize_filename(filename):
    """
    Standardizes the filename: removes problematic characters and replaces spaces with underscores.
    """
    # 1. Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # 2. Remove specified characters: comma, single quote, dash, parenthesis
    filename = re.sub(r"[',\-(),]", "", filename)
    # Convert to lowercase to prevent case issues
    filename = filename.lower()
    return filename


def get_manual_candidate_details(uploaded_cvs, job_id=None):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.debug(f"Created/verified folder: {UPLOAD_FOLDER}")

    # delete files in the current folder (CRITICAL for cleanup before saving new files)
    try:
        folder_path = UPLOAD_FOLDER
        files = glob.glob(os.path.join(folder_path, '*'))
        for file_path in files:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.debug(f"Removed {file_path}")
    except Exception as e:
        # Note: Corrected the static log message "downloaded_cvs" to the variable folder_path
        logger.warning(f"Could not delete temporary files in {folder_path}: {str(e)}")
    
    candidate_details = {}
    
    # save files in the folder and populate candidate_details with UUIDs
    for idx, cv in enumerate(uploaded_cvs):
        
        # 1. Sanitize the filename
        sanitized_name = sanitize_filename(cv.filename)
        file_location = os.path.join(UPLOAD_FOLDER, sanitized_name)
        
        # 2. Generate a unique candidate ID (UUID)
        # Use a combination of UUID and offset to make the ID somewhat traceable 
        # (Though UUID is globally unique by itself)
        candidate_uuid = str(uuid.uuid4()) + str(idx)
        
        try:
            with open(file_location, "wb") as buffer:
                # IMPORTANT: Reset the file pointer before copying ass a FastAPI UploadFile/SpooledTemporaryFile
                cv.file.seek(0) 
                shutil.copyfileobj(cv.file, buffer)
            logger.info(f"Successfully saved cv {sanitized_name} with UUID {candidate_uuid}")
            
            # Store the UUID as the key and the local file path as the 'cv' value
            candidate_details[candidate_uuid] = {
                'cv': file_location, # This is the full path to the saved file
                'email': "",         # Email is unavailable for manual uploads
            }
            
        except Exception as e:
            logger.error(f"Couldn't save cvs to file {str(e)}", exc_info=True)
    
    if candidate_details:
        logger.info(f"Returning details for {len(candidate_details)} manually uploaded candidates.")
        return candidate_details
    else:
        logger.error("No CV files were successfully saved or processed.")
        return {}


# Function to download CVs
def download_cvs(candidates_details, folder_name="downloaded_cvs"):
    logger.info(f"Starting CV download process for {len(candidates_details)} candidates")
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    logger.debug(f"Created/verified folder: {folder_name}")
    
    # Dictionary to store candidate_id -> filename mapping
    cv_files_map = {}
    successful_downloads = 0
    failed_downloads = 0
    failed_downloads_links = []

    for candidate_id, details in candidates_details.items():
        
        if not details['cv']:
            logger.warning(f"No CV found for candidate {candidate_id}")
            continue
        try:
            link = BASE_URL + '/' + details['cv'] + SAS_TOKEN
            logger.debug(f"Downloading CV for candidate {candidate_id}")
            
            response = requests.get(link, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                # Detect original file extension from CV link
                original_ext = os.path.splitext(details['cv'])[1].lower()

                # Fallback to .pdf if extension is missing
                if original_ext not in [".pdf", ".docx", ".doc"]:
                    original_ext = ".pdf"

                filename = f"cv_{candidate_id}{original_ext}"

                filepath = os.path.join(folder_name, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                cv_files_map[candidate_id] = {
                    'email': candidates_details.get(candidate_id, {}).get('email', ''),
                    'cv': filepath,
                    }
                successful_downloads += 1
                logger.info(f"Successfully downloaded CV {filename} for candidate {candidate_id}")
            else:
                failed_downloads += 1
                failed_downloads_links.append(link)
                logger.error(f"Failed to download CV for candidate {candidate_id}. Status code: {response.status_code}")
        except Exception as e:
            failed_downloads += 1
            failed_downloads_links.append(link)
            logger.error(f"Error downloading CV for candidate {candidate_id}: {str(e)}", exc_info=True)
    
    logger.info(f"Download process completed. Success: {successful_downloads}, Failed: {failed_downloads}")
    return cv_files_map, failed_downloads_links


# Function to download CVs
def download_cvs_manual(candidates_details, folder_name="downloaded_cvs_manual"):
    logger.info(f"Starting CV download process for {len(candidates_details)} candidates")
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    logger.debug(f"Created/verified folder: {folder_name}")
    
    # Dictionary to store candidate_id -> filename mapping
    cv_files_map = {}
    successful_downloads = 0
    failed_downloads = 0
    failed_downloads_links = []

    for candidate_id, details in candidates_details.items():
        if not details['url']:
            logger.warning(f"No CV found for candidate {candidate_id}")
            continue
        try:
            link = details['url'] + SAS_TOKEN
            print(f"link is {link}")
            logger.debug(f"Downloading CV for candidate {candidate_id}")
            
            response = requests.get(link, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                # Detect original file extension from CV link
                original_ext = os.path.splitext(details['url'])[1].lower()

                # Fallback to .pdf if extension is missing
                if original_ext not in [".pdf", ".docx", ".doc"]:
                    original_ext = ".pdf"

                filename = f"cv_{candidate_id}{original_ext}"
                print(f"filename is {filename}")

                filepath = os.path.join(folder_name, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                cv_files_map[candidate_id] = {
                    'email': '',
                    'cv': filepath,
                    }
                successful_downloads += 1
                logger.info(f"Successfully downloaded CV {filename} for candidate {candidate_id}")
            else:
                failed_downloads += 1
                failed_downloads_links.append(link)
                logger.error(f"Failed to download CV for candidate {candidate_id}. Status code: {response.status_code}")
        except Exception as e:
            failed_downloads += 1
            failed_downloads_links.append(link)
            print(f"Error downloading CV for candidate {candidate_id}: {str(e)}")
            logger.error(f"Error downloading CV for candidate {candidate_id}: {str(e)}", exc_info=True)
    
    logger.info(f"Download process completed. Success: {successful_downloads}, Failed: {failed_downloads}")
    return cv_files_map, failed_downloads_links


# Function to process the CV files that have already been downloaded/uploaded
def process_cv_folder(candidates_details, db_manager, job_id):
    """
    Processes all CV files using paths stored in candidates_details, 
    extracts text, and handles failures gracefully.
    
    Args:
        candidates_details (dict): A dictionary mapping candidate_id (str) 
                                   to details: {'cv': filepath (str), 'email': email (str)}.
                                   The 'cv' value must be the local file path.
        db_manager (DatabaseManager): An instance of the database manager.
        job_id (int): The ID of the job being processed.

    Returns:
        tuple: A tuple containing two lists (processed_candidates, failed_candidates).
    """
    logger.info(f"Starting CV processing for {len(candidates_details)} candidates.")
    processed_candidates = []
    failed_candidates = []

    for candidate_id, details in candidates_details.items():
       
        file_path = details['cv']
        
        if not os.path.exists(file_path):
            failure_reason = "File not found during processing."
            logger.error(f"CV for candidate {candidate_id} not found at {file_path}. Reason: {failure_reason}")
            failed_candidates.append({
                'candidate_id': candidate_id,
                'reason': failure_reason
            })
            # Note: Since the file was never downloaded/uploaded successfully, we don't save a failed record here
            # or rely on cleanup, but log it as a download/upload failure.
            continue
            
        logger.debug(f"Processing CV for candidate {candidate_id} at {file_path}")
        
        try:
            # Extract text
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == ".docx":
                text = extract_text_from_docx(file_path)
            elif ext == ".doc":
                text = extract_text_from_doc(file_path)
            else:
                raise Exception(f"Unsupported file type: {ext}")
            
            # Create candidate data dictionary
            candidate_data = {
                'candidate_id': candidate_id,
                'email': candidates_details[candidate_id]['email'],
                'cv_text': text,
                'cv_path': file_path
            }

            processed_candidates.append(candidate_data)
            logger.info(f"Successfully processed CV for candidate {candidate_id}")
            
        except Exception as e:
            failure_reason = f"Error during text extraction: {str(e)}"
            logger.error(f"Error processing CV for candidate {candidate_id}: {failure_reason}", exc_info=True)
            failed_candidates.append({
                'candidate_id': candidate_id,
                'reason': failure_reason
            })
            # Save the failed candidate details to the database
            db_manager.save_failed_candidate(job_id=job_id, candidate_id=candidate_id, reason=failure_reason, filepath=file_path)

        finally:
            # Delete the file after processing, whether successful or not
            try:
                os.remove(file_path)
                logger.debug(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {file_path}: {str(e)}")

    logger.info(f"Processing completed. Success: {len(processed_candidates)}, Failed: {len(failed_candidates)}")
    return processed_candidates, failed_candidates