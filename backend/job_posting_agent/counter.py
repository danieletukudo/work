from fastapi import Depends, HTTPException
from azure.cosmos import CosmosClient
import os, uuid
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv()

# --- Setup Cosmos Client ---
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "RecruitmentDB"
CONTAINER_NAME = "Employers"

client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# --- Helper: Check Employer Limit ---
def check_and_update_generation_limit(employer_id: str, job_title: str):
    employer = container.read_item(item=employer_id, partition_key=employer_id)

    if employer["generation_count"] >= 3:
        raise HTTPException(
            status_code=403,
            detail="Job description generation limit reached (max 3)."
        )

    # Update employer record
    employer["generation_count"] += 1
    employer["job_generations"].append({
        "timestamp": datetime.now(UTC).isoformat(),
        "job_title": job_title
    })

    container.upsert_item(employer)
    return employer["generation_count"]
