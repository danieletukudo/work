import mysql.connector
from mysql.connector import IntegrityError
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
from cv_agent.loggerfile import setup_logging
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
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    job_id VARCHAR(100) PRIMARY KEY,
                    total_cvs INT,
                    auto_processed_cvs INT DEFAULT 0,
                    manual_processed_cvs INT DEFAULT 0,
                    status VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cost (
                    job_id VARCHAR(100) PRIMARY KEY,
                    cost FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Table analysis_cost created successfully")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_id VARCHAR(100),
                    candidate_id VARCHAR(100),
                    name VARCHAR(100),
                    user_email VARCHAR(100),
                    platform_email VARCHAR(100),
                    fitness_score FLOAT,
                    analysis_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_job_candidate (job_id, candidate_id)
                )
            """)
            print("Table candidates created successfully")

            # Create failed_candidates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failed_candidates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    job_id VARCHAR(100),
                    candidate_id VARCHAR(100),
                    reason TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("Table failed_candidates created successfully")

            # create table for cv filrs data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cv_files (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100),
                    blob_name TEXT,
                    url TEXT,
                    file_hash TEXT,
                    file_size TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("Table cv_files table created successfully")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cv_meta (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(100),
                    blob_name TEXT,
                    url TEXT,
                    file_size TEXT,
                    file_hash TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("Table cv_files table created successfully")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS HR_USERS (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(100) UNIQUE,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("Table HR_USERS table created successfully")

            self.conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}", exc_info=True)
            raise
    
    # check if a file with the same hash exists for a job
    async def file_exists(self, job_id: str, file_hash: str):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM cv_files WHERE job_id = %s AND file_hash = %s",
                (job_id, file_hash)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking candidate existence for job {job_id} {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()

    # get the list of all exisiting hashes for a job
    async def get_existing_file_hashes(self, job_id: str) -> set:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT file_hash FROM cv_files WHERE job_id = %s",
                (job_id,)
            )
            rows = cursor.fetchall()
            return {row[0] for row in rows}
        except Exception as e:
            logger.error(f"Error fetching existing file hashes for job {job_id}: {str(e)}", exc_info=True)
            return set()
        finally:
            cursor.close()
    
    
    # def update_job_status(self, job_id: str, total_cvs: int, processed_cvs: int, status: str, source: str):
    #     """
    #     source: should be either "auto" or "manual"
    #     processed_cvs: the new processed count for that source.
    #     """
    #     cursor = self.conn.cursor()
    #     try:
    #         if source == "auto":
    #             cursor.execute("""
    #                 INSERT INTO analysis_jobs (job_id, total_cvs, auto_processed_cvs, manual_processed_cvs, status)
    #                 VALUES (%s, %s, %s, %s, %s)
    #                 ON DUPLICATE KEY UPDATE
    #                 auto_processed_cvs = %s,
    #                 status = %s
    #             """, (job_id, total_cvs, processed_cvs, 0, status, processed_cvs, status))
    #         elif source == "manual":
    #             cursor.execute("""
    #                 INSERT INTO analysis_jobs (job_id, total_cvs, auto_processed_cvs, manual_processed_cvs, status)
    #                 VALUES (%s, %s, %s, %s, %s)
    #                 ON DUPLICATE KEY UPDATE
    #                 manual_processed_cvs = %s,
    #                 status = %s
    #             """, (job_id, total_cvs, 0, processed_cvs, status, processed_cvs, status))
    #         else:
    #             raise ValueError("Invalid source. Must be 'auto' or 'manual'.")
    #         print("Job status updated successfully")
    #         self.conn.commit()
    #     finally:
    #         cursor.close()


    def update_job_status(self, job_id: str, total_cvs: int = None, processed_cvs: int = 0, status: str = None, source: str = "auto"):
        """
        Update job row:
        - If source == "auto", increment auto_processed_cvs by processed_increment.
        - If source == "manual", increment manual_processed_cvs by processed_increment.
        - Optionally set total_cvs and status if provided.
        This uses an atomic SQL update to avoid race conditions.
        """
        processed_increment = processed_cvs
        cursor = self.conn.cursor()
        try:
            # Ensure row exists
            cursor.execute("""
                INSERT INTO analysis_jobs (job_id, total_cvs, auto_processed_cvs, manual_processed_cvs, status)
                VALUES (%s, COALESCE(%s, 0), 0, 0, COALESCE(%s, 'queued'))
                ON DUPLICATE KEY UPDATE
                total_cvs = COALESCE(%s, total_cvs)
            """, (job_id, total_cvs, status, total_cvs))
            # Atomic increment for the appropriate column
            if processed_increment and processed_increment != 0:
                if source == "auto":
                    cursor.execute("""
                        UPDATE analysis_jobs
                        SET auto_processed_cvs = auto_processed_cvs + %s,
                            status = COALESCE(%s, status),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (processed_increment, status, job_id))
                elif source == "manual":
                    cursor.execute("""
                        UPDATE analysis_jobs
                        SET manual_processed_cvs = manual_processed_cvs + %s,
                            status = COALESCE(%s, status),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (processed_increment, status, job_id))
                else:
                    raise ValueError("source must be 'auto' or 'manual'")
            else:
                # Only update status if no increment requested
                if status:
                    cursor.execute("""
                        UPDATE analysis_jobs
                        SET status = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = %s
                    """, (status, job_id))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def candidate_exists(self, job_id: str, platform_email: str) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT candidate_id FROM candidates WHERE job_id = %s AND platform_email = %s",
                (job_id, platform_email)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking candidate existence for job {job_id} and email {platform_email}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()
    
    async def save_file_metadata(self, job_id: str, uploaded_files: list):
        """
        Stores uploaded file info in DB.
        """
        cursor = self.conn.cursor()

        try:
            for f in uploaded_files:
                cursor.execute("""
                    INSERT INTO cv_files (job_id, original_filename, blob_name, url, file_hash)
                    VALUES (%s, %s, %s, %s, %s)
                """, ( 
                    job_id,
                    f["original_filename"],
                    f["blob_name"],
                    f["url"],
                    f["file_hash"]
                    ))
                
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving file metadata for job {job_id}: {str(e)}", exc_info=True)
        finally:
            cursor.close()
    
    async def save_cv_metadata(self, job_id: str, uploaded_files: list):
        """
        Stores uploaded file info in DB.
        """
        cursor = self.conn.cursor()

        try:
            for f in uploaded_files:
                cursor.execute("""
                    INSERT INTO cv_files (job_id, blob_name, url, file_hash, file_size)
                    VALUES (%s, %s, %s, %s, %s)
                """, ( 
                    job_id,
                    f["blob_name"],
                    f["url"],
                    f["file_hash"],
                    f["file_size"]
                    ))
                
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving cv metadata for job {job_id}: {str(e)}", exc_info=True)
        finally:
            cursor.close()

    # delete cvs metadata for a job
    async def delete_cv_metadata(self, job_id: str) -> bool:
        """
    Cleans up CV metadata from the database for a given job ID.
    """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM cv_files WHERE job_id = %s",
                (job_id,)
            )
            self.conn.commit()
            print(f"Metadata for job {job_id} successfully deleted from the DB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting CV metadata for job {job_id}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()
    
    def fetch_hr_users(self) -> List[Dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM HR_USERS")
            return cursor.fetchall()
        finally:
            cursor.close()
    
    # get available cvs for a job
    def get_candidate_files(self, job_id: str) -> List[Dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM cv_files WHERE job_id = %s ORDER BY uploaded_at DESC",
                (job_id,)
            )
            return cursor.fetchall()
        finally:
            cursor.close()

    def save_candidate(self, job_id: str, candidate) -> bool:
        cursor = self.conn.cursor()
        try:
            # Check if a candidate with the same user_email exists for the job
            cursor.execute("""
                SELECT candidate_id FROM candidates 
                WHERE job_id = %s AND user_email = %s
                """, (job_id, candidate.email))
            existing = cursor.fetchall()

            if existing:
                print(f"Candidate {candidate.email} already exists for job {job_id}. Updating record.")
                logger.debug(f"Candidate {candidate.email} already exists for job {job_id}. Updating record.")
                # Candidate exists, update its record (replace old candidate with the new one)
                cursor.execute("""
                    UPDATE candidates 
                    SET candidate_id = %s,
                        name = %s,
                        platform_email = %s,
                        fitness_score = %s,
                        analysis_summary = %s
                    WHERE job_id = %s AND user_email = %s
                """, (
                    candidate.candidate_id,
                    candidate.name,
                    candidate.platform_email,
                    candidate.fitness_score,
                    candidate.analysis_summary,
                    job_id,
                    candidate.email
                ))
            else:
                # No candidate with the same email exists, so insert a new record
                cursor.execute("""
                    INSERT INTO candidates (
                        job_id,
                        candidate_id,
                        name,
                        user_email,
                        platform_email,
                        fitness_score,
                        analysis_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    
                    job_id,
                    candidate.candidate_id,
                    candidate.name,
                    candidate.email,
                    candidate.platform_email,
                    candidate.fitness_score,
                    candidate.analysis_summary
                ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving candidate {candidate.candidate_id} for job {job_id}: {str(e)}", exc_info=True)
            return False
        finally:
            cursor.close()

    def save_failed_candidate(self, job_id: str, candidate_id: str, reason: str, filepath: str) -> None:
        cursor = self.conn.cursor()
        try:
            # Check if a failed candidate record already exists for this job and candidate
            cursor.execute(
                "SELECT reason FROM failed_candidates WHERE job_id = %s AND candidate_id = %s",
                (job_id, candidate_id)
            )
            existing = cursor.fetchone()
            if existing is not None:
                # Concatenate the new reason with the existing one
                existing_reason = existing[0]
                new_reason = f"{existing_reason}; {reason}"
                cursor.execute(
                    "UPDATE failed_candidates SET reason = %s, file_path = %s, created_at = CURRENT_TIMESTAMP WHERE job_id = %s AND candidate_id = %s",
                    (new_reason, filepath, job_id, candidate_id)
                )
            else:
                # Insert a new failed candidate record with file_path
                cursor.execute(
                    "INSERT INTO failed_candidates (job_id, candidate_id, reason, file_path) VALUES (%s, %s, %s, %s)",
                    (job_id, candidate_id, reason, filepath)
                )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving failed candidate {candidate_id} for job {job_id}: {str(e)}", exc_info=True)
        finally:
            cursor.close()

    def get_failed_candidates(self, job_id: str) -> list:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute(
                "SELECT * FROM failed_candidates WHERE job_id = %s ORDER BY created_at DESC",
                (job_id,)
            )
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def save_analysis_cost(self, job_id: str, cost: float):
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO analysis_cost (job_id, cost)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE
                cost = cost + VALUES(cost)
            """, (job_id, cost))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving analysis cost for job {job_id}: {str(e)}", exc_info=True)
        finally:
            cursor.close()

    def get_top_candidates(self, job_id: str, limit: Optional[int] = None) -> List[Dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            if limit:
                query = """
                    SELECT * FROM candidates 
                    WHERE job_id = %s 
                    ORDER BY fitness_score DESC 
                    LIMIT %s
                """
                cursor.execute(query, (job_id, limit))
            else:
                query = """
                    SELECT * FROM candidates 
                    WHERE job_id = %s 
                    ORDER BY fitness_score DESC
                """
                cursor.execute(query, (job_id,))
            return cursor.fetchall()
        finally:
            cursor.close()

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT * FROM analysis_jobs 
                WHERE job_id = %s
            """, (job_id,))
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def get_analysis_cost(self, job_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT * FROM analysis_cost 
                WHERE job_id = %s
            """, (job_id,))
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def get_candidate_result(self, job_id: str, candidate_id: str) -> Optional[Dict]:
        """
        Fetches a single candidate's analysis result by job_id and candidate_id.
        Returns a dictionary or None if not found.
        """
        cursor = self.conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT * FROM candidates 
                WHERE job_id = %s AND candidate_id = %s
            """, (job_id, candidate_id))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error fetching candidate {candidate_id} for job {job_id}: {str(e)}",
                         exc_info=True)
            return None
        finally:
            cursor.close()

    def close(self):
        self.conn.close()

    