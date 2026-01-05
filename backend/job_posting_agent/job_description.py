import os
from fastapi import FastAPI
import google.generativeai as genai
import json
from pydantic import ValidationError
import os
from dotenv import load_dotenv
from google.genai import types
from google import genai
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .loggerfile import setup_logging
from .read_examples import read_job_description_example

logger = setup_logging("job_posting_agent")
load_dotenv()

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

# --- Configuration ---
class GeminiConfig:
    """Configuration settings for Gemini API and retry mechanism."""
    SYSTEM_PROMPT = """You are an expert HR and recruitment assistant. Your task
    is to generate a detailed and compelling job description based on the following information.
    The job description should be professional, well-structured, and suitable for a public job board.
    Do not include any conversational text outside of the job description itself.
    the details provided by the employer as as follows
        """
    MODEL_NAME = 'gemini-2.0-flash-001'
    TEMPERATURE = 0.7
    TOP_P = 0.8
    MAX_RETRIES = 4
    RETRY_DELAY = 2  # seconds
    RETRY_BACKOFF = 2  # multiplicative factor for delay
    MAX_WORKERS_THREAD_POOL = 5 # For CPU-bound tasks in CandidateAnalyzer

app = FastAPI(
    title="AI Job Description Generator",
    description="A backend to generate job descriptions using the Gemini API."
)


class JobDesscriptionGenerator:
    """
    Handles interactions with the Gemini API to generate job description.
    Uses a ThreadPoolExecutor for CPU-bound tasks if necessary, though Gemini API calls are I/O bound.
    The primary use of ThreadPoolExecutor here is to wrap synchronous API calls in an async context
    if the underlying Gemini client does not fully support async.
    """
    def __init__(self):
        load_dotenv()
        self.gemini_key = os.getenv("JOB_DESCRIPTION_GEMINI_KEY")
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


    async def generate_job_description(self, prompt: str):
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
                    prompt
                )
                
                if response and hasattr(response, 'candidates') and response.candidates[0].content.parts[0].text:
                    # Ensure the parsed response conforms to CvDetails
                    try:
                        # Convert parsed object to dictionary then to CvDetails
                        output_text = response.candidates[0].content.parts[0].text
                        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
                        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                        logger.info("job description generated successfully.")
                        return output_text, input_tokens, output_tokens
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

    def _get_gemini_response(self, prompt: str) -> types.GenerateContentResponse:
        """  
        Synchronous call to the Gemini API to generate content.
        This method is designed to be run in a ThreadPoolExecutor.
        """
        try:
            additional_prompt = "/n" + "follow the structure of the example provided below: " + read_job_description_example()
            response = self.gemini_client.models.generate_content(
                model=GeminiConfig.MODEL_NAME,
                contents=f"here are the details provided by the employer: {prompt}",
                config=types.GenerateContentConfig(
                    system_instruction= GeminiConfig.SYSTEM_PROMPT + additional_prompt,
                    temperature=GeminiConfig.TEMPERATURE,
                    top_p=GeminiConfig.TOP_P,
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
            if not (response, 'candidates') and response.candidates[0].content.parts[0].text:
                logger.error(f"Gemini API returned a response without a 'parsed' attribute or with empty parsed content. Response: {response}")
                raise GeminiAPIError("Gemini API returned an unparseable or empty response.")

            return response
        except Exception as e:
            logger.error(f"Error during Gemini API call in _get_gemini_response: {str(e)}", exc_info=True)
            raise GeminiAPIError(f"Gemini API call failed: {str(e)}") from e
