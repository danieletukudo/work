from fastapi import APIRouter, UploadFile, File, Form
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueClient
import os, uuid, json

router = APIRouter()

AZURE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("CV_CONTAINER")
QUEUE = os.getenv("QUEUE_NAME")
 
blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
queue_client = QueueClient.from_connection_string(AZURE_CONN, QUEUE)


@router.post("/upload_cvs")
async def upload_cvs(
    files: list[UploadFile] = File(...),
    job_id: str = Form(...)
):
    container = blob_service.get_container_client(CONTAINER)
    container.create_container()

    uploaded_files = []

    for file in files:
        ext = file.filename.split(".")[-1]
        new_name = f"{uuid.uuid4()}.{ext}"

        blob = container.get_blob_client(new_name)
        blob.upload_blob(await file.read())

        url = blob.url
        uploaded_files.append({
            "filename": file.filename,
            "blob_name": new_name,
            "url": url
        })

    # Send queue message
    msg = {
        "job_id": job_id,
        "files": uploaded_files
    }
    queue_client.send_message(json.dumps(msg))

    return {
        "status": "queued",
        "job_id": job_id,
        "file_count": len(uploaded_files)
    }
