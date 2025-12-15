from azure.storage.blob.aio import BlobServiceClient
from uuid import uuid4
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER_NAME)


async def save_to_azure_blob(files, job_id: str, db):
    """
    Uploads multiple CV files to Azure Blob Storage.
    Returns a list of uploaded file metadata.
    """
    uploaded_files = []

    for file in files:
        # Read file bytes
        content = await file.read()

        # Compute MD5 hash
        file_hash = hashlib.md5(content).hexdigest()

        # Check if file with same hash already exists for this job
        if await db.file_exists(job_id, file_hash):
            print(f"Skipping duplicate file: {file.filename}")
            continue

        extension = file.filename.split(".")[-1]
        blob_name = f"{job_id}/{uuid4()}.{extension}"

        blob_client = container_client.get_blob_client(blob_name)
        # content = await file.read()

        await blob_client.upload_blob(content, overwrite=True)

        uploaded_files.append({
            "original_filename": file.filename,
            "blob_name": blob_name,
            "url": blob_client.url,
            "file_hash": file_hash
        })

    return uploaded_files
