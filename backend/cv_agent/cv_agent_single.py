import json
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
import os
from dotenv import load_dotenv
from google.genai import types
from google import genai
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import time
from .prompts import prompts
from cv_agent.loggerfile import setup_logging

load_dotenv()

logger = setup_logging("single_cv_analyzer")

# --- Configuration ---
class GeminiConfig:
    """Configuration settings for Gemini API and retry mechanism."""
    SYSTEM_PROMPT = prompts
    MODEL_NAME = 'gemini-2.5-flash'
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_RETRIES = 4
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # multiplicative factor for delay
    MAX_WORKERS_THREAD_POOL = 5 # For CPU-bound tasks in CandidateAnalyzer

# --- Exceptions ---
class CVAnalyzerException(Exception):
    """Base exception for CV analysis errors."""
    pass

class GeminiAPIError(CVAnalyzerException):
    """Exception for errors specifically from the Gemini API."""
    pass

class DataProcessingError(CVAnalyzerException):
    """Exception for errors during data parsing or validation."""
    pass

# --- Decorators ---
def retry_with_backoff(func: Callable) -> Callable:
    """
    A decorator to retry an asynchronous function with exponential backoff.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        retries = 0
        delay = GeminiConfig.RETRY_DELAY

        while retries < GeminiConfig.MAX_RETRIES:
            try:
                return await func(*args, **kwargs)
            except (GeminiAPIError, CVAnalyzerException) as e: # Catch specific exceptions
                retries += 1
                if retries == GeminiConfig.MAX_RETRIES:
                    logger.error(f"Failed after {retries} retries: {str(e)}", exc_info=True)
                    raise CVAnalyzerException(f"Failed after {retries} retries: {str(e)}") from e # Chain exceptions

                wait_time = delay * (GeminiConfig.RETRY_BACKOFF ** (retries - 1))
                logger.warning(f"Attempt {retries} for {func.__name__} failed. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e: # Catch any other unexpected errors
                logger.critical(f"An unexpected error occurred in {func.__name__}: {str(e)}", exc_info=True)
                raise CVAnalyzerException(f"An unexpected error occurred: {str(e)}") from e
        # This part should ideally not be reached if MAX_RETRIES is hit and an exception is raised
        logger.error(f"Function {func.__name__} completed without returning a value after retries, indicating a logical error.")
        raise CVAnalyzerException(f"Function {func.__name__} failed to return a value after retries.")
    return wrapper

# --- Pydantic Models ---
class CvDetails(BaseModel):
    """Pydantic model for parsed CV details and analysis."""
    analysis_summary: Optional[str]
    fitness_score: float

    class Config:
        # Allow extra fields to be ignored if Gemini returns something unexpected,
        # but it's generally better to strictly define the schema.
        # For production, consider if strictness is preferred to catch unexpected outputs.
        extra = "ignore"


# --- Core Logic Classes ---
class CandidateAnalyzer:
    """
    Handles interactions with the Gemini API to analyze CVs.
    Uses a ThreadPoolExecutor for CPU-bound tasks if necessary, though Gemini API calls are I/O bound.
    The primary use of ThreadPoolExecutor here is to wrap synchronous API calls in an async context
    if the underlying Gemini client does not fully support async.
    """
    def __init__(self):
        self.gemini_key = os.getenv("CV_ANALYSIS_GEMINI_KEY")
        if not self.gemini_key:
            logger.error("Gemini API key not found in environment variables.")
            raise CVAnalyzerException("Gemini API key not found in environment variables.")
        
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_key)

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client or load model: {str(e)}", exc_info=True)
            raise CVAnalyzerException(f"Failed to initialize Gemini client: {str(e)}") from e

        # Use ThreadPoolExecutor for potentially blocking operations within _get_gemini_response
        # This is crucial if `generate_content` is a blocking synchronous call.
        self._executor = ThreadPoolExecutor(max_workers=GeminiConfig.MAX_WORKERS_THREAD_POOL)
        logger.info("Gemini client and ThreadPoolExecutor initialized successfully.")

    @retry_with_backoff
    async def analyze_cv(self, additional: str, cv_text: str, jd_text: str) -> CvDetails:
        """
        Analyzes a single CV against a job description using the Gemini API.
        Applies retry logic with exponential backoff.
        """
        logger.info(f"Starting CV analysis for a candidate.")
        try:
            # Run the potentially blocking _get_gemini_response in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._get_gemini_response,
                additional,
                cv_text,
                jd_text
            )
            
            if response and hasattr(response, 'parsed') and response.parsed:
                # Ensure the parsed response conforms to CvDetails
                try:
                    # Convert parsed object to dictionary then to CvDetails
                    gemini_output_dict = json.loads(response.parsed.model_dump_json())
                    input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
                    output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                    cv_details = CvDetails(**gemini_output_dict)
                    logger.info("CV analysis completed successfully.")
                    return cv_details, input_tokens, output_tokens
                except ValidationError as ve:
                    logger.error(f"Pydantic validation error for Gemini response: {ve}", exc_info=True)
                    raise DataProcessingError(f"Invalid data returned by Gemini API: {ve}") from ve
                except json.JSONDecodeError as jde:
                    logger.error(f"JSON decode error when parsing Gemini response: {jde}", exc_info=True)
                    raise DataProcessingError(f"Invalid JSON in Gemini response: {jde}") from jde
                except Exception as e:
                    logger.error(f"Unexpected error when processing Gemini response: {e}", exc_info=True)
                    raise DataProcessingError(f"Failed to process Gemini response: {e}") from e
            else:
                logger.error("Gemini API returned an empty or invalid response.")
                raise GeminiAPIError("Gemini API returned an empty or invalid response.")

        except GeminiAPIError:
            raise # Re-raise if it's a Gemini specific error already handled by retry
        except CVAnalyzerException:
            raise # Re-raise custom exceptions
        except Exception as e:
            logger.error(f"Error in analyze_cv: {str(e)}", exc_info=True)
            raise CVAnalyzerException(f"Error analyzing CV: {str(e)}") from e

    def _get_gemini_response(self, additional: str, cv_text: str, jd_text: str) -> types.GenerateContentResponse:
        """
        Synchronous call to the Gemini API to generate content.
        This method is designed to be run in a ThreadPoolExecutor.
        """
        try:
            response = self.gemini_client.models.generate_content(
                model=GeminiConfig.MODEL_NAME,
                contents=f"Explicitly follow the instructions given in {additional}",
                config=types.GenerateContentConfig(
                    system_instruction=GeminiConfig.SYSTEM_PROMPT + f"the cv text is: {cv_text} and job description text is {jd_text}",
                    response_mime_type='application/json',
                    response_schema=CvDetails,
                    temperature=GeminiConfig.TEMPERATURE,
                    top_p=GeminiConfig.TOP_P
                ),
            )
            # Check for potential errors in the response object itself
            if response.candidates and response.candidates[0].finish_reason != types.FinishReason.STOP:
                logger.warning(f"Gemini response finished with reason: {response.candidates[0].finish_reason.name}")
                if response.candidates[0].safety_ratings:
                    for rating in response.candidates[0].safety_ratings:
                        if rating.probability > types.HarmProbability.NEGLIGIBLE:
                            logger.warning(f"Safety issue detected: {rating.category.name} with probability {rating.probability.name}")
                # Decide if this should be an error or just a warning based on your tolerance
                # For a recruitment agent, it's safer to treat non-STOP reasons as potential issues.
                raise GeminiAPIError(f"Gemini API did not complete generation cleanly. Finish reason: {response.candidates[0].finish_reason.name}")
            
            # Additional check for actual content
            if not (hasattr(response, 'parsed') and response.parsed):
                logger.error(f"Gemini API returned a response without a 'parsed' attribute or with empty parsed content. Response: {response}")
                raise GeminiAPIError("Gemini API returned an unparseable or empty response.")

            return response
        except Exception as e:
            logger.error(f"Error during Gemini API call in _get_gemini_response: {str(e)}", exc_info=True)
            raise GeminiAPIError(f"Gemini API call failed: {str(e)}") from e


class CostCalculator:
    """Calculates the cost of API usage based on input and output tokens."""
    def __init__(self, input_cost_per_million_tokens: float, output_cost_per_million_tokens: float):
        # Costs are per million tokens
        self.input_cost = input_cost_per_million_tokens
        self.output_cost = output_cost_per_million_tokens

    def calculate_total_cost(self, total_input_token: int, total_output_token: int) -> float:
        """Calculates the total cost based on token counts."""
        input_price = (self.input_cost * total_input_token) / 1_000_000
        output_price = (self.output_cost * total_output_token) / 1_000_000
        return input_price + output_price


# class CVProcessor:
#     """Orchestrates the CV analysis process, including candidate processing and cost calculation."""
#     def __init__(self):
#         self.analyzer = CandidateAnalyzer()
#         # Use more precise token costs if available for your Gemini model
#         # These are illustrative values. Check actual Gemini pricing.
#         self.total_input_token = 0
#         self.total_output_token = 0
#         self.cost_calculator = CostCalculator(input_cost_per_million_tokens=0.001, output_cost_per_million_tokens=0.002)
#         self.semaphore = asyncio.Semaphore(GeminiConfig.MAX_WORKERS_THREAD_POOL) # Control concurrent API calls
#         logger.info("CVProcessor initialized successfully.")

#     async def process_candidates(
#         self,
#         processed_candidates: List[Dict],
#         jd_text: str,
#         instruction: str = "",
#         db: Any = None,
#         job_id: Any = None
#     ) -> Tuple[List[CvDetails], float]:
#         """
#         Processes a list of candidates asynchronously and calculates the total cost.
#         """
#         tasks = []
#         all_candidate_results: List[CvDetails] = []

#         logger.info(f"Starting to process {len(processed_candidates)} candidates.")


#         async def analyze_and_collect(candidate_data: Dict):
#             """Helper async function to analyze a single candidate and collect results."""
#             print("checking for existing candidate in db")
#             if db and job_id and candidate_data.get("email"):
#                 if db.candidate_exists(str(job_id), candidate_data.get("email")):
#                     print(f"""Candidate {candidate_data.get('candidate_id', 'N/A')} with platform email {candidate_data.get('email')} already exists.
#                              Skipping analysis.""")
#                     logger.info(f"Candidate {candidate_data.get('candidate_id', 'N/A')} with platform email {candidate_data.get('email')} already exists. Skipping analysis.")
#                     return
    
#             async with self.semaphore: # Limit concurrent API calls
#                 try:
#                     # Pass the instruction as 'additional' to the analyzer
#                     cv_details_result, input_tokens, output_tokens = await self.analyzer.analyze_cv(
#                         additional=instruction,
#                         cv_text=candidate_data['cv_text'],
#                         jd_text=jd_text
#                     )

#                     # Update platform_email and candidate_id from the original candidate_data
#                     cv_details_result.platform_email = candidate_data.get('email')
#                     cv_details_result.candidate_id = candidate_data.get('candidate_id')
#                     if "@" not in candidate_data.get('email', '') and "@" not in cv_details_result.platform_email:
#                         cv_details_result.platform_email = cv_details_result.email  # Fallback to email if platform_email is not valid
         
#                     # Collect token counts for cost calculation
#                     self.total_input_token += input_tokens
#                     self.total_output_token += output_tokens
                    
#                     all_candidate_results.append(cv_details_result)
#                     logger.debug(f"Processed candidate {candidate_data.get('candidate_id', 'N/A')} successfully.")
#                 except CVAnalyzerException as e:
#                     db.save_failed_candidate(
#                         job_id=job_id,
#                         candidate_id=candidate_data.get('candidate_id', 'N/A'),
#                         reason=str(e),
#                         filepath=candidate_data.get('cv_path', 'N/A')  # Save file path if available
#                     ) if db else logger.error(f"Failed to save candidate {candidate_data.get('candidate_id', 'N/A')} due to missing DB.")

#                 except Exception as e:
#                     logger.critical(f"""Unexpected error during candidate processing for
#                                     {candidate_data.get('candidate_id', 'N/A')}: {e}""", exc_info=True)
#                 finally:
#                     # wait a bit to avoid overwhelming the API
#                     await asyncio.sleep(1)

#         for candidate in processed_candidates:
#             tasks.append(analyze_and_collect(candidate))

#         await asyncio.gather(*tasks)
        
#         # Placeholder: To estimate cost without direct token counts:
#         # avg_input_tokens_per_cv = 500 # Estimate
#         # avg_output_tokens_per_cv = 100 # Estimate
#         # total_input_token_sum = len(all_candidate_results) * avg_input_tokens_per_cv
#         # total_output_token_sum = len(all_candidate_results) * avg_output_tokens_per_cv

#         total_cost = self.cost_calculator.calculate_total_cost(self.total_input_token, self.total_output_token)
#         logger.info(f"Completed processing {len(all_candidate_results)} candidates. Total cost: ${total_cost:.4f}")
#         return all_candidate_results, total_cost