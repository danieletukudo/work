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
from .database import DatabaseManager
from .loggerfile import setup_logging

logger = setup_logging("personality_agent")

db = DatabaseManager()

personality_types = """Openness to experience: How curious and creative a person is.
                    Conscientiousness: How organized and dependable a person is.
                    Extraversion: How outgoing and sociable a person is.
                    Agreeableness: How cooperative and empathetic a person is.
                    Neuroticism (also called Emotional Stability): How emotionally
                    stable and resilient a person is.
                    """

# --- Configuration ---
class GeminiConfig:
    """Configuration settings for Gemini API and retry mechanism."""
    SYSTEM_PROMPT = f"""
    You are a personality scoring assistant which is a part of
    a recruitment automation system, the employer has given personality
    preferences for ideal candidates in description text and traits dictionary

    the expected output is a JSON with target_profile containing
    each of the big five personality traits with expected value and weight.

    Output JSON:
    {{
      "target_profile": {{
        "Openness": {{"expected": float, "weight": float}},
        "Conscientiousness": {{"expected": float, "weight": float}},
        "Extraversion": {{"expected": float, "weight": float}},
        "Agreeableness": {{"expected": float, "weight": float}},
        "Neuroticism": {{"expected": float, "weight": float}}
      }}
    }}
    Ensure weights sum to 1.
    """
    MODEL_NAME = 'gemini-2.0-flash-001'
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_RETRIES = 4
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # multiplicative factor for delay
    MAX_WORKERS_THREAD_POOL = 5 # For CPU-bound tasks in CandidateAnalyzer

# --- Exceptions ---
class PersonalityException(Exception):
    """Base exception for CV analysis errors."""
    pass

class GeminiAPIError(PersonalityException):
    """Exception for errors specifically from the Gemini API."""
    pass

class DataProcessingError(PersonalityException):
    """Exception for errors during data parsing or validation."""
    pass


# --- Pydantic Models ---
class CvDetails(BaseModel):
    """Pydantic model for parsed CV details and analysis."""
    name: str
    email: str
    phone: str
    analysis_summary: Optional[str]
    fitness_score: float
    platform_email: Optional[str] = None
    candidate_id: Optional[Union[str, int]] = None

    class Config:
        # Allow extra fields to be ignored if Gemini returns something unexpected,
        # but it's generally better to strictly define the schema.
        # For production, consider if strictness is preferred to catch unexpected outputs.
        extra = "ignore"


# --- Core Logic Classes ---
class PersonalityTypeGenerator:
    """
    Handles interactions with the Gemini API to generate the personality types.
    Uses a ThreadPoolExecutor for CPU-bound tasks if necessary, though Gemini API calls are I/O bound.
    The primary use of ThreadPoolExecutor here is to wrap synchronous API calls in an async context
    if the underlying Gemini client does not fully support async.
    """
    def __init__(self):
        load_dotenv()
        self.gemini_key = os.getenv("GEMINI_KEY")
        if not self.gemini_key:
            logger.error("Gemini API key not found in environment variables.")
            raise PersonalityException("Gemini API key not found in environment variables.")
        
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_key)

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client or load model: {str(e)}", exc_info=True)
            raise PersonalityException(f"Failed to initialize Gemini client: {str(e)}") from e

        # Use ThreadPoolExecutor for potentially blocking operations within _get_gemini_response
        # This is crucial if `generate_content` is a blocking synchronous call.
        self._executor = ThreadPoolExecutor(max_workers=GeminiConfig.MAX_WORKERS_THREAD_POOL)
        logger.info("Gemini client and ThreadPoolExecutor initialized successfully.")

    async def generate_personality(self, description: str, normalized_traits: dict):
        """
        generate personality preference from employers input using the Gemini API.
        Applies retry logic with exponential backoff.
        """
        logger.info(f"Starting preference generation for a job.")
        try:
            # Run the potentially blocking _get_gemini_response in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._get_gemini_response,
                description,
                normalized_traits
            )
            
            if response and hasattr(response, 'candidates') and response.candidates[0].content.parts[0].text:
                # Ensure the parsed response conforms to CvDetails
                try:
                    # Convert parsed object to dictionary then to CvDetails
                    response_dict = json.loads(response.candidates[0].content.parts[0].text)
                    logger.info("personality preference generated successfully.")
                    return response_dict
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
        except PersonalityException:
            raise # Re-raise custom exceptions
        except Exception as e:
            logger.error(f"Error in peronsality generation: {str(e)}", exc_info=True)
            raise PersonalityException(f"Error generating personality by agent: {str(e)}") from e

    def _get_gemini_response(self, description: str, normalized_traits: dict) -> types.GenerateContentResponse:
        """
        Synchronous call to the Gemini API to generate content.
        This method is designed to be run in a ThreadPoolExecutor.
        """
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=f"the description text is: {description} and traits are {normalized_traits}",
                config=types.GenerateContentConfig(
                    system_instruction=GeminiConfig.SYSTEM_PROMPT,
                    response_mime_type='application/json',
                    temperature=0.8,
                    top_p=0.5
                ),
            )

            # Check for potential errors in the response object itself
            if response.candidates and response.candidates[0].finish_reason != types.FinishReason.STOP:
                logger.warning(f"Gemini response finished with reason: {response.candidates[0].finish_reason.name}")
                if response.candidates[0].safety_ratings:
                    for rating in response.candidates[0].safety_ratings:
                        if rating.probability > types.HarmProbability.NEGLIGIBLE:
                            logger.warning(f"Safety issue detected: {rating.category.name} with probability {rating.probability.name}")
                # For a recruitment agent, it's safer to treat non-STOP reasons as potential issues.
                raise GeminiAPIError(f"Gemini API did not complete generation cleanly. Finish reason: {response.candidates[0].finish_reason.name}")
            
            # Additional check for actual content
            if not (response, 'candidates') and response.candidates[0].content.parts[0].text:
                logger.error(f"Gemini API returned a response without a 'parsed' attribute or with empty parsed content. Response: {response}")
                raise GeminiAPIError("Gemini API returned an unparseable or empty response.")

            return response
        except Exception as e:
            logger.error(f"Error during Gemini API call in _get_gemini_response: {str(e)}", exc_info=True)
            raise GeminiAPIError(f"Gemini API call failed: {str(e)}") from e


# 2. Normalize slider values
def normalize_traits(traits_dict):
    return {k: v / 100 for k, v in traits_dict.items()}


# 3. Generate weights using Gemini
# def generate_weights_with_gemini(description, normalized_traits):
#     model = GenerativeModel("gemini-pro")  # Ensure SDK is configured
#     prompt = f"""
#     You are a personality scoring assistant. Given the description:
#     "{description}"
#     and the following trait importance (0-1 scale):
#     {normalized_traits}

#     Output JSON:
#     {{
#       "target_profile": {{
#         "Openness": {{"expected": float, "weight": float}},
#         "Conscientiousness": {{"expected": float, "weight": float}},
#         "Extraversion": {{"expected": float, "weight": float}},
#         "Agreeableness": {{"expected": float, "weight": float}},
#         "Neuroticism": {{"expected": float, "weight": float}}
#       }}
#     }}
#     Ensure weights sum to 1.
#     """
#     response = model.generate_content(prompt)
#     return json.loads(response.text)

# 4. Full pipeline

personality_generator = PersonalityTypeGenerator()

async def build_target_personality(job_posting_id: str) -> dict:
    try:
        results = db.get_personality_preferences(job_posting_id)
        if not results:
            return {"error": "Employer preferences not found"}
        traits = results[0]["traits"]
        description = results[0]["description"]
        traits_new = json.loads(traits)
    except Exception as e:
        logger.error(f"Error fetching personality preferences for job_posting_id {job_posting_id}: {str(e)}", exc_info=True)
        return {"error": f"Error fetching personality preferences: {str(e)}"}
    try:
        normalized_traits = normalize_traits(traits_new)
        target_profile = await personality_generator.generate_personality(description, normalized_traits)
        if target_profile:
            save_result = db.save_target_personality(job_posting_id, target_profile)
            print(save_result)
    except PersonalityException as pe:
        logger.error(f"Personality generation error for job_posting_id {job_posting_id}: {str(pe)}", exc_info=True)
        return {"error": f"Personality generation error: {str(pe)}"}
    return target_profile