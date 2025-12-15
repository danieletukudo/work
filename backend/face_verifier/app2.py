import os
import faiss
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import cv2
from .loggerfile import setup_logging
import os

logger = setup_logging("face_verifier_v2_app")

class FaceDB:
    def __init__(self, db_path="faiss_index.bin", mapping_path="id_mapping.pkl", dim=512, use_gpu=False):
        self.db_path = db_path
        self.mapping_path = mapping_path
        self.dim = dim

        provider = 'CUDAExecutionProvider' if use_gpu else 'CPUExecutionProvider'
        self.app = FaceAnalysis(name="buffalo_l", providers=[provider])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Cosine similarity FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_mapping = {}

        if os.path.exists(self.db_path) and os.path.exists(self.mapping_path):
            self._load_index()
            print("Database loaded.")
            logger.info("Database loaded.")
        else:
            print("No existing database found. Starting fresh.")
            logger.info("No existing database found. Starting fresh.")

    def _normalize(self, emb):
        return emb / np.linalg.norm(emb)
    
    
    def _preprocess_image(self, image_path: str):
        """
        Read, enhance, and convert image to RGB format.
        """
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Error: Unable to load image from path: {image_path}"
            raise ValueError(msg)

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance low-light images (Histogram Equalization on Y channel)
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        logger.info(f"Image {image_path} preprocessed successfully")
        return enhanced

    def _get_embedding(self, image_path):
        img = self._preprocess_image(image_path)
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return self._normalize(faces[0].embedding.astype("float32"))

    def enroll_user(self, user_id, image_paths: dict):
        """
        Enroll user with multiple angles.
        image_paths = { "front": path, "left": path, ... }
        """
        embeddings = []
        for angle, path in image_paths.items():
            emb = self._get_embedding(path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return False, f"No valid faces found for user {user_id}"

        start_idx = self.index.ntotal
        self.index.add(np.array(embeddings))

        # record mapping
        for i in range(len(embeddings)):
            self.id_mapping[start_idx + i] = user_id

        self._save_index()
        print(f"User {user_id} enrolled with {len(embeddings)} embeddings")
        logger.info(f"User {user_id} enrolled with {len(embeddings)} embeddings")
        return True, f"User {user_id} enrolled with {len(embeddings)} embeddings"

    def verify_user(self, user_id, image_paths: list, threshold=0.5):
        """
        Verify by checking if *any* verification image matches enrolled user.
        """
        if len(self.id_mapping) == 0:
            return False, "Database empty"
        msgs = []
        for path in image_paths:
            emb = self._get_embedding(path)
            if emb is None:
                logger.warning(f"No face found in image {path}, skipping...")
                msgs.append(f"No face detected")
                continue

            sims, idxs = self.index.search(np.expand_dims(emb, axis=0), k=5)
            sims, idxs = sims[0], idxs[0]

            # check if any retrieved embedding belongs to user_id
            for sim, idx in zip(sims, idxs):
                if idx == -1:
                    continue
                if self.id_mapping.get(idx) == user_id and sim >= threshold:
                    print(f"Match found for user {user_id} with score {sim:.2f}")
                    logger.info(f"Match found for user {user_id} with score {sim:.2f}")
                    return True, "match"
        
        if len(msgs) == len(image_paths):
            return False, "no match - No face detected in images"
        
        logger.info(f"No match found for user {user_id}")
        print(f"No match found for user {user_id}")
        return False, "no match"

    def _save_index(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.id_mapping, f)

    def _load_index(self):
        self.index = faiss.read_index(self.db_path)
        with open(self.mapping_path, "rb") as f:
            self.id_mapping = pickle.load(f)
