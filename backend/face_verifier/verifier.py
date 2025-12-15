import os
import time
import base64
import json
import requests
from pathlib import Path
import os
from typing import Any
from .loggerfile import setup_logging
from dotenv import load_dotenv
from azure.storage.queue import QueueServiceClient
from .app2 import FaceDB
from pathlib import Path
import requests

# Load environment
load_dotenv()

logger = setup_logging("face_verifier_queue_processor")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
QUEUE_NAME = os.getenv("QUEUE_NAME")
DEAD_LETTER_QUEUE_NAME = f"{QUEUE_NAME}-deadletter"
DOWNLOAD_FOLDER = Path(os.getenv("DOWNLOAD_FOLDER", "downloaded"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RECEIVING_URL = os.getenv("RECEIVING_URL")
IMAGES_BASE_URL = os.getenv("IMAGES_BASE_URL")
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

headers = {
        "User-Agent": os.getenv("USER_AGENT"),
        "Referer": os.getenv("REFERER"),
        "Accept": os.getenv("ACCEPT"),
        "Accept-Language": os.getenv("ACCEPT_LANGUAGE"),
        "Connection": "keep-alive"
        }

# Queue clients
queue_service_client = QueueServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

try:
    queue_client = queue_service_client.get_queue_client(QUEUE_NAME)
    dead_letter_queue_client = queue_service_client.get_queue_client(DEAD_LETTER_QUEUE_NAME)
except Exception as e:
    logger.error(f"Failed to get queue clients: {e}")
    raise e

db = FaceDB(use_gpu=False)


def download_image(url: str, filename: str) -> bool:
    """Download image from URL and save to disk"""
    try:
        # url = url + os.getenv("SAS_TOKEN")
        resp = requests.get(url, headers=headers,
                            allow_redirects=True,
                            stream=True,
                            timeout=15)
        resp.raise_for_status()
        file_path = DOWNLOAD_FOLDER / filename
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        logger.error(f"Failed to download {url}: {e}")
        return False

def send_result(batch: str, user_id: str, status: str, captured_time: Any = None):
    """Send verification result to external API"""
    payload = {
        "batch": batch,
        "user_id": str(user_id),
        "status": status,
        "captured_time": captured_time
    }

    print(f"Sending result to API: {payload}")
    try:
        resp = requests.post(
            RECEIVING_URL,
            json=payload,
            timeout=10,
            headers=headers,
            allow_redirects=True
        )
        resp.raise_for_status()
        logger.info(f"✅ Result delivered to API for user {user_id}, batch {batch}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to deliver result for {user_id}, batch {batch}: {e}")
        return False


def process_queue():
    """Continuously poll Azure Queue and process messages"""
    print(" Queue processor started...")


    while True:
        # receive messages
        messages = queue_client.receive_messages(messages_per_page=5, visibility_timeout=30)
        received = False

        for msg_batch in messages.by_page():
            
            for msg in msg_batch:
                received = True

                # Decode message (base64 -> JSON)
                content = base64.b64decode(msg.content).decode("utf-8")
                data1 = json.loads(content)
                # print(f"📬 Received message: {records}")
                data = data1["records"]

                print(data)
                if len(data) == 0:
                    print(" No records found in message, deleting...")
                    logger.warning("No records found in message, deleting...")
                    queue_client.delete_message(msg)
                    continue
                
                user_id = data[0].get('user_id')
                print(f"User ID: {user_id}")
                batch = data[0].get('batch')
                print(f"Batch: {batch}")
                captured_time = data[2].get('captured_time')
                print(f"Captured Time: {captured_time}")
                
                enroll_record = {
                    "front": data[0]["front"],
                    "left": data[0]["left"],
                    "right": data[0]["right"],
                    "faceup": data[0]["faceup"],
                    "facedown": data[0]["facedown"]
                }

                print(f"📥 Processing image for user {user_id}, batch {batch}")
                logger.info(f"Processing image for user {user_id}, batch {batch}")
                # print("Image URL:", ["imgpath"])
                image_paths = []

                if user_id in db.id_mapping.values():
                    print(f"User {user_id} already enrolled.")
                
                else:
                    enroll_images = {}
                    for keys, value in enroll_record.items():
                        filename = f"{batch}_{user_id}_{keys}.jpg"
                        print(f"filename is: {filename}")
                        success = download_image(value, filename)
                        enroll_images[keys] = DOWNLOAD_FOLDER / filename if success else None
                    db.enroll_user(user_id, enroll_images)

                # file_name1 = f"{batch}_{user_id}_{data[0]['image_name']}"
                # base = IMAGES_BASE_URL
                # url1 = base + data[0]["imgpath"]
                # download_image(url1, file_name1)
                # image_paths.append(DOWNLOAD_FOLDER / file_name1)

                for record in data[1:]:
                    filename = f"{batch}_{user_id}_{record['image_name']}"
                    print(f"filename is: {filename}")
                    success = download_image(record["imgpath"], filename)
                    image_paths.append(DOWNLOAD_FOLDER / filename if success else None)
                if success:
                    print(f"✅ Saved images in record successfully")
                    # Process images
                    is_match, msg = db.verify_user(user_id, image_paths[1:], threshold=0.5)
                    print(is_match, msg)

                    # known_image_path = image_paths[0]
                    # face_verifier = FaceVerifier(known_image_path)
                    # result = face_verifier.verify_faces(image_paths[1:])
                    logger.info(f"Images processed for batch {batch}, user {user_id} successfully")
                    print(f"images processed for batch {batch}, user {user_id} successfully")
                    print(f"Result: {msg}")

                    # Send to webhook
                    send_result(batch, user_id, msg, captured_time)

                    # Delete images after processing
                    # Delete images after analysis
                    images_paths = os.listdir(DOWNLOAD_FOLDER)
                    images_paths = [os.path.join(DOWNLOAD_FOLDER, img) for img in images_paths
                                   if img.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    for images in images_paths:
                        try:
                            os.remove(images)
                            logger.info(f"Deleted image: {images}")
                        except Exception as e:
                            logger.warning(f"Could not delete image {images}: {e}")

                    queue_client.delete_message(msg)
                else:
                    print(f"⚠️ Failed processing {filename}, dequeue count: {msg.dequeue_count}")

                    if msg.dequeue_count >= MAX_RETRIES:
                        logger.warning(f"💀 Message failed {msg.dequeue_count} times, moving to dead-letter queue")
                        print(f"💀 Message failed {msg.dequeue_count} times, moving to dead-letter queue")

                        # Push to dead-letter queue
                        dead_letter_queue_client.send_message(msg.content)

                        # Remove from main queue
                        queue_client.delete_message(msg)

        if not received:
            # wait for 10 seconds before polling again
            time.sleep(2)


if __name__ == "__main__":
    try:
        process_queue()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully")
