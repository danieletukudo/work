import cv2
import face_recognition
from .loggerfile import setup_logging
import os


logger = setup_logging("face_verifier_app")


class FaceRecognition:
    """
    Face Recognition Class for comparing faces in images.
    """

    def __init__(self, known_image_path: str, tolerance: float = 0.6):
        """
        Initialize with a known image and tolerance for face matching.
        """
        self.known_image_path = known_image_path
        self.tolerance = tolerance
        self.known_encoding = self._get_face_encoding(known_image_path)

    def _preprocess_image(self, image_path: str):
        """
        Read and convert image to RGB format.
        """
        image = cv2.imread(image_path)
        if image is None:
            msg = f"Error: Unable to load image from path: {image_path}"
            logger.error(msg)
            raise ValueError(msg)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _get_face_encoding(self, image_path: str):
        """
        Get face encoding from an image.
        """
        try:
            image = self._preprocess_image(image_path)
            encodings = face_recognition.face_encodings(image)
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

        if not encodings:
            msg = f"No face detected in image: {image_path}"
            logger.error(msg)
            return "no face detected"

        return encodings[0]

    def compare_faces(self, unknown_image_path: str):
        """
        Compare an unknown image with the known image.
        Returns (True, message) if a match is found,
        otherwise (False, message).
        """
        try:
            if isinstance(self.known_encoding, str) and self.known_encoding == "no face detected":
                return False, "No face detected"
            unknown_encoding = self._get_face_encoding(unknown_image_path)
            if isinstance(unknown_encoding, str) and unknown_encoding == "no face detected":
                return False, "No face detected"
            results = face_recognition.compare_faces([self.known_encoding],
                                                     unknown_encoding,
                                                     tolerance=self.tolerance)
            distance = face_recognition.face_distance([self.known_encoding], unknown_encoding)[0]
            confidence = 1 - distance

            if results[0]:
                msg = f"Match found! Confidence: {confidence:.2%}"
                logger.info(msg)
                return True, msg
            else:
                msg = f"No match. Similarity: {confidence:.2%}"
                logger.info(msg)
                return False, msg
        except Exception as e:
            logger.error(f"Error comparing face in {unknown_image_path}: {e}")
            return False, str(e)


class FaceVerifier:
    """
    Handles face verification across multiple images.
    """

    def __init__(self, known_image_path: str):
        self.face_recognition = FaceRecognition(known_image_path)

    def _select_images(self, image_paths: list[str]) -> list[str]:
        """
        Select images for verification. Uses first, last, and three evenly spaced images.
        """
        if not image_paths:
            msg = "No images found."
            logger.error(msg)
            raise ValueError(msg)

        if len(image_paths) < 3:
            logger.info("Fewer than 3 images available; using all images.")
            return image_paths

        indices = [0.25, 0.5, 0.75]
        selected_images = [image_paths[0], image_paths[-1]]
        selected_images.extend([image_paths[int(i * len(image_paths))]
                                for i in indices])
        logger.info(f"Selected images for verification: {selected_images}")
        return selected_images

    def verify_faces(self, image_paths: list[str]) -> str:
        """
        Compare selected images with the known image and determine if a match exists.
        """
        try:
            selected_images = self._select_images(image_paths)
            results = [self.face_recognition.compare_faces(img)[0] for img in selected_images]
            msgs = [self.face_recognition.compare_faces(img)[1] for img in selected_images]
            print("msgs:", msgs)
            verdict = "match" if any(results) else "no match"
            if verdict == "no match" and "No face detected" in msgs:
                verdict = "no match - some images had no detectable faces"
            logger.info(f"Verification verdict: {verdict}")

            # Delete images after analysis
            for img_path in image_paths:
                try:
                    os.remove(img_path)
                    logger.info(f"Deleted image: {img_path}")
                except Exception as e:
                    logger.warning(f"Could not delete image {img_path}: {e}")


            return verdict
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            return "Could not match faces at the moment"