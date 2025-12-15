import mysql.connector
from mysql.connector import IntegrityError
from typing import List, Dict, Optional
from dotenv import load_dotenv
import json
import os
from .loggerfile import setup_logging
load_dotenv()


logger = setup_logging("database")

class DatabaseManager:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )
            self._create_tables()
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}", exc_info=True)
            raise

    def _create_tables(self):
        try:
            cursor = self.conn.cursor()
            # Drop existing candidates table
            # cursor.execute("DROP TABLE IF EXISTS candidates")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_preferences (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_posting_id VARCHAR(100),
                    traits JSON,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidate_personality (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    candidate_id VARCHAR(100) UNIQUE,
                    personality_profile JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create target_personalities table to store the output of build_target_personality
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS target_personalities (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_posting_id VARCHAR(100),
                    target_profile JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Tables created or verified successfully.")
            self.conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}", exc_info=True)
            raise
    
    def save_personality_preferences(self, job_posting_id: str, traits: dict, description: str) -> bool:
        cursor = self.conn.cursor()
        try:
            traits_json = json.dumps(traits)
            cursor.execute("""
                INSERT INTO personality_preferences (job_posting_id, traits, description)
                VALUES (%s, %s, %s)
            """, (job_posting_id, traits_json, description))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving personality preferences for job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()
    
    def save_target_personality(self, job_posting_id: str, target_profile: dict) -> bool:
        cursor = self.conn.cursor()
        try:
            profile_json = json.dumps(target_profile)
            cursor.execute("""
                INSERT INTO target_personalities (job_posting_id, target_profile)
                VALUES (%s, %s)
            """, (job_posting_id, profile_json))
            self.conn.commit()
            print(f"Target personality saved for job_posting_id {job_posting_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving target personality for job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()

    def get_personality_preferences(self, job_posting_id: str) -> list:
        cursor = self.conn.cursor(dictionary=True)
        try:
            query = """
                SELECT * FROM personality_preferences
                WHERE job_posting_id = %s
                ORDER BY created_at DESC
            """
            cursor.execute(query, (job_posting_id,))
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error(f"Error retrieving personality preferences for job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
            return []
        finally:
            cursor.close()
    
    def get_target_personality(self, job_posting_id: str) -> list:
        cursor = self.conn.cursor(dictionary=True)
        try:
            query = """
                SELECT * FROM target_personalities
                WHERE job_posting_id = %s
                ORDER BY created_at DESC
            """
            cursor.execute(query, (job_posting_id,))
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.error(f"Error retrieving target personality for job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
            return []
        finally:
            cursor.close()

    def save_candidate_personality(self, candidate_id: str, personality_profile: dict) -> bool:
        cursor = self.conn.cursor()
        try:
            profile_json = json.dumps(personality_profile)
            cursor.execute("""
                INSERT INTO candidate_personality (candidate_id, personality_profile)
                VALUES (%s, %s)
            """, (candidate_id, profile_json))
            self.conn.commit()
            return True
        except IntegrityError:
            logger.warning(f"Candidate with candidate_id {candidate_id} already exists.")
            return False
        except Exception as e:
            logger.error(f"Error saving candidate personality for candidate_id {candidate_id}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()

    def get_candidate_personality(self, candidate_id: str) -> Optional[dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            query = """
                SELECT * FROM candidate_personality
                WHERE candidate_id = %s
            """
            cursor.execute(query, (candidate_id,))
            result = cursor.fetchone()
            return result
        except Exception as e:
            logger.error(f"Error retrieving candidate personality for candidate_id {candidate_id}: {str(e)}", exc_info=True)
            return None
        finally:
            cursor.close()

    def close(self):
        self.conn.close()
