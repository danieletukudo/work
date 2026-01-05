from pymongo import MongoClient
from datetime import datetime, UTC
import os
import urllib.parse
from dotenv import load_dotenv
from .loggerfile import setup_logging

load_dotenv()

logger = setup_logging("job_posting_db")

class MongoDBConnection2:
    def __init__(self):
        self.root = os.getenv("COSMOS_ROOT")
        self.tail = os.getenv("COSMOS_TAIL")
        self.username = urllib.parse.quote_plus(os.getenv("COSMOS_USER"))
        self.password = urllib.parse.quote_plus(os.getenv("COSMOS_KEY"))
        self.uri = f"{self.root}{self.username}:{self.password}{self.tail}"
        # print(self.uri)
        self.client = MongoClient(self.uri)
        self.db = self.client.get_database(os.getenv("MONGODB_NAME"))
    
    # funtion to create a new collection if not exists
    def create_collection(self, collection_name):
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created.")
        else:
            logger.info("Collection '{collection_name}' already exists.")

    def get_job_posting_collection(self):
        self.create_collection("job_postings")
        return self.db.job_postings


class AdminDatabase:
    def __init__(self):
        self.connection = MongoDBConnection2()

    def add_job_posting(self, job_description_data):
        job_descriptions_collection = self.connection.get_job_posting_collection()
        result = job_descriptions_collection.insert_one(job_description_data)
        return str(result.inserted_id)

    def get_job_posting(self, job_id):
        job_descriptions_collection = self.connection.get_job_posting_collection()
        return job_descriptions_collection.find_one({"job_id": job_id})

    def append_conversation(self, job_id, role, content):
        job_descriptions_collection = self.connection.get_job_posting_collection()
        update_result = job_descriptions_collection.update_one(
            {"job_id": job_id},
            {
                "$push": {
                    "conversation_history": {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "role": role,
                        "content": content
                    }
                }
            }
        )
        return update_result.modified_count

    def increment_generation_count(self, job_id):
        job_descriptions_collection = self.connection.get_job_posting_collection()
        job_descriptions_collection.update_one(
            {"job_id": job_id},
            {"$inc": {"generation_count": 1}, "$set": {"updated_at": datetime.now(UTC).isoformat()}}
        )
    
    def delete_job_posting(self, job_id):
        job_posting_collection = self.connection.get_job_posting_collection()
        delete_result = job_posting_collection.delete_one({"job_id": job_id})
        return delete_result.deleted_count

    def delete_all_job_postings(self):
        job_posting_collection = self.connection.get_job_posting_collection()
        delete_result = job_posting_collection.delete_many({})
        return delete_result.deleted_count