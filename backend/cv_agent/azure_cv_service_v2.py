from azure.storage.blob.aio import BlobServiceClient
import os
from datetime import datetime, timedelta, timezone
from azure.storage.blob import generate_container_sas, BlobSasPermissions
from dotenv import load_dotenv
from .database import DatabaseManager

load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    raise EnvironmentError(
        "Missing required environment variables: AZURE_STORAGE_CONNECTION_STRING or AZURE_CONTAINER_NAME"
    )

try: 
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)
except Exception as e:
    print(f"Error initializing Azure Blob Service Client: {e}")
    # In a real app, this should log and potentially fail startup
    blob_service_client = None
    container_client = None


def generate_sas_token(job_id: str):
    """
    Generates a SAS token allowing write access to the specific container.
    """
    # Define permissions: Write and Create only (no delete or read for security
    # if possible, though read is needed to check existence sometimes)
    
    permissions = BlobSasPermissions(write=True, create=True, read=True)
    
    expiry_time = datetime.now(timezone.utc) + timedelta(hours=1)

    sas_token = generate_container_sas(
        account_name=blob_service.account_name,
        container_name=CONTAINER_NAME,
        account_key=blob_service.credential.account_key, # Ensure you are using account key auth, not Managed Identity for this specific method
        permission=permissions,
        expiry=expiry_time
    )

    return {
        "sas_token": sas_token,
        "container_name": CONTAINER_NAME,
        "account_name": blob_service.account_name,
        "expiry": expiry_time
    }


def clean_database_metadata(job_id: str) -> bool:
    """
    Placeholder function to delete all job-related metadata from the database.
    
    This function should connect to your database (e.g., MongoDB, PostgreSQL, CosmosDB)
    and delete all records where the 'job_id' matches the input.

    Args:
        job_id: The ID of the job whose metadata should be cleared.

    Returns:
        True if deletion was successful, False otherwise.
    """
    db = DatabaseManager()
    print(f"--- Database Cleanup for Job ID: {job_id} ---")
    result = db.delete_cv_metadata(job_id)
    if not result:
        print(f"Failed to delete metadata for job {job_id} from the DB.")
        return False
    print(f"Metadata for job {job_id} successfully marked for deletion in the DB.")
    return True

async def delete_job_blobs_from_azure(job_id: str):
    """
    Deletes all blobs (CV files) associated with a given job ID.
    The job_id is used as the virtual folder prefix.
    """
    if not container_client:
        raise Exception("Azure Container Client not initialized.")
        
    prefix = f"{job_id}/"
    print(f"Searching for blobs with prefix: {prefix}")
    
    blobs_to_delete = []
    
    # 1. List all blobs starting with the job ID prefix
    # Use name_starts_with to efficiently query only the relevant blobs
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    
    for blob in blob_list:
        blobs_to_delete.append(blob.name)

    if not blobs_to_delete:
        print(f"No CV files found for Job ID: {job_id}")
        return 0

    print(f"Found {len(blobs_to_delete)} files to delete.")

    # 2. Delete the blobs in batches (up to 256 blobs per call for efficiency)
    deleted_count = 0
    batch_size = 256
    
    for i in range(0, len(blobs_to_delete), batch_size):
        batch = blobs_to_delete[i:i + batch_size]
        try:
            # delete_blobs() can delete up to 256 blobs in a single request
            container_client.delete_blobs(*batch)
            deleted_count += len(batch)
            print(f"Successfully deleted batch of {len(batch)} blobs.")
        except Exception as e:
            # Log the error but continue trying to delete other batches
            print(f"Error deleting a batch of blobs: {e}")

    return deleted_count

# from azure.storage.blob import BlobServiceClient, CorsRule
# import os
# from dotenv import load_dotenv

# load_dotenv()

# AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# def configure_cors():
#     blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

#     # Define the CORS rule
#     cors_rule = CorsRule(
#         allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Add production URL later
#         allowed_methods=["PUT", "POST", "GET", "OPTIONS", "HEAD"],
#         allowed_headers=["*"],
#         exposed_headers=["*"],
#         max_age_in_seconds=3600
#     )

#     # Set properties for the Blob service
#     blob_service.set_service_properties(cors=[cors_rule])
#     print("CORS rules updated successfully for Azure Blob Storage.")

# if __name__ == "__main__":
#     configure_cors()