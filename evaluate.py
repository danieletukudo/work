from typing import List, Dict, Optional
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
import os
import aiohttp
import asyncio

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Evaluation API endpoint
EVALUATION_API_ENDPOINT = "https://genesis-rw-ai.remoting.work/analyze-single-cv"

def evaluate_interview_comprehensive(
    job_description: Dict,
    candidate_cv: Dict,
    candidate_info: Dict,
    interview_transcript: Dict,
    evaluation_instruction: Optional[str] = None,
    use_api: bool = False
) -> Dict:
    """
    Comprehensive evaluation using job description, CV, candidate info, and interview transcript.
    
    Args:
        job_description: Dictionary containing job details (title, department, location, etc.)
        candidate_cv: Dictionary containing candidate CV data (education, experience, skills, etc.)
        candidate_info: Dictionary containing candidate_id, candidate_email, candidate_name, job_id
        interview_transcript: Dictionary containing interview transcript (session_history, real_time_transcript)
        evaluation_instruction: Optional custom instruction for evaluation
        use_api: If True, send evaluation to external API endpoint
    
    Returns:
        Dict: Comprehensive evaluation including scores and feedback
    """
    from prompt import get_evaluation_instruction, prepare_api_payload
    
    # Generate evaluation prompt
    evaluation_prompt = get_evaluation_instruction(
        job_description=job_description,
        candidate_cv=candidate_cv,
        candidate_info=candidate_info,
        interview_transcript=interview_transcript,
        evaluation_instruction=evaluation_instruction
    )
    
    # If using external API, prepare payload
    if use_api:
        try:
            payload = prepare_api_payload(
                job_description=job_description,
                candidate_cv=candidate_cv,
                candidate_info=candidate_info,
                interview_transcript=interview_transcript,
                evaluation_instruction=evaluation_instruction
            )
            
            # Make async API call
            async def call_api():
                async with aiohttp.ClientSession() as session:
                    async with session.post(EVALUATION_API_ENDPOINT, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"API call failed with status {response.status}: {error_text}")
            
            # Run async call
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                import nest_asyncio
                nest_asyncio.apply()
                result = asyncio.run(call_api())
            else:
                result = loop.run_until_complete(call_api())
            
            return result
            
        except Exception as e:
            print(f" API call failed: {e}")
            print("Falling back to local OpenAI evaluation...")
            # Fall through to local evaluation
    
    # Local evaluation using OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}  # Request JSON format
        )

        raw_response = response.choices[0].message.content

        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            evaluation = json.loads(raw_response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown or text
            try:
                # Look for JSON between triple backticks
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content between curly braces
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON found in response")
            except:
                print("\nDebug - Raw response from API:")
                print(raw_response)
                raise

        # Validate and set default values for required fields
        required_fields = {
            "overall_score": 5,
            "technical_competency": 5,
            "problem_solving": 5,
            "communication": 5,
            "experience_level": 5,
            "cultural_fit": 5,
            "cv_alignment": 5,
            "interview_performance": 5,
            "strengths": [],
            "areas_for_improvement": [],
            "cv_vs_interview_consistency": "medium",
            "hiring_recommendation": "maybe",
            "detailed_feedback": "Evaluation completed",
            "individual_question_scores": [],
            "key_highlights": [],
            "skill_assessment": {},
            "recommendations": {}
        }

        for field, default_value in required_fields.items():
            if field not in evaluation:
                evaluation[field] = default_value

        # Ensure candidate info is included
        evaluation["candidate_id"] = candidate_info.get("candidate_id", "N/A")
        evaluation["candidate_name"] = candidate_info.get("candidate_name", "N/A")
        evaluation["job_id"] = candidate_info.get("job_id", "N/A")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        # Return a default evaluation structure
        evaluation = {
            "candidate_id": candidate_info.get("candidate_id", "N/A"),
            "candidate_name": candidate_info.get("candidate_name", "N/A"),
            "job_id": candidate_info.get("job_id", "N/A"),
            "overall_score": 5,
            "technical_competency": 5,
            "problem_solving": 5,
            "communication": 5,
            "experience_level": 5,
            "cultural_fit": 5,
            "cv_alignment": 5,
            "interview_performance": 5,
            "strengths": ["Unable to evaluate"],
            "areas_for_improvement": ["Unable to evaluate"],
            "cv_vs_interview_consistency": "medium",
            "hiring_recommendation": "maybe",
            "detailed_feedback": f"Error occurred during evaluation: {str(e)}",
            "individual_question_scores": [],
            "key_highlights": [],
            "skill_assessment": {},
            "recommendations": {
                "hire_decision": "maybe",
                "reasoning": "Evaluation error occurred",
                "next_steps": [],
                "onboarding_notes": ""
            }
        }

    return evaluation


def evaluate_interview(job_role: str, interview_data: Dict) -> Dict:
    """
    Evaluate candidate responses and generate comprehensive feedback.

    Args:
        job_role (str): The position being interviewed for
        interview_data (Dict): JSON data with job_transcript containing Q&A pairs

    Returns:
        Dict: Structured evaluation including scores and feedback
    """
    from prompt import get_simple_evaluation_prompt
    
    # Extract Q&A pairs from the JSON structure
    qa_pairs = interview_data.get("job_transcript", [])
    
    # Get evaluation prompt from prompt.py
    prompt = get_simple_evaluation_prompt(job_role, qa_pairs)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        raw_response = response.choices[0].message.content

        # Try to extract JSON from the response
        try:
            # First, try direct JSON parsing
            evaluation = json.loads(raw_response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown or text
            try:
                # Look for JSON between triple backticks
                json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content between curly braces
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON found in response")
            except:
                print("\nDebug - Raw response from API:")
                print(raw_response)
                raise

        # Validate required fields
        required_fields = [
            "overall_score", "technical_competency", "problem_solving",
            "communication", "experience_level", "cultural_fit",
            "strengths", "areas_for_improvement", "hiring_recommendation",
            "detailed_feedback", "individual_question_scores"
        ]

        for field in required_fields:
            if field not in evaluation:
                if field == "overall_score":
                    evaluation[field] = 5
                elif field == "individual_question_scores":
                    evaluation[field] = []
                else:
                    evaluation[field] = "Not provided"

    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Return a default evaluation structure
        evaluation = {
            "overall_score": 5,
            "technical_competency": 5,
            "problem_solving": 5,
            "communication": 5,
            "experience_level": 5,
            "cultural_fit": 5,
            "strengths": ["Unable to evaluate"],
            "areas_for_improvement": ["Unable to evaluate"],
            "hiring_recommendation": "maybe",
            "detailed_feedback": f"Error occurred during evaluation: {str(e)}",
            "individual_question_scores": []
        }
    except:
        pass

    return evaluation


def evaluate_from_files(
    job_description_path: Optional[str] = None,
    candidate_cv_path: Optional[str] = None,
    candidate_info_path: Optional[str] = None,
    interview_transcript_path: str = None,
    evaluation_instruction: Optional[str] = None,
    use_api: bool = False
) -> Dict:
    """
    Load all data from files and run comprehensive evaluation.
    
    Args:
        job_description_path: Path to job description JSON file
        candidate_cv_path: Path to candidate CV JSON file
        candidate_info_path: Path to candidate info JSON file
        interview_transcript_path: Path to interview transcript JSON file
        evaluation_instruction: Optional custom instruction
        use_api: If True, use external API endpoint
    
    Returns:
        Dict: Evaluation results
    """
    from prompt import load_evaluation_data_from_files
    
    # Load data from files using prompt.py function
    data = load_evaluation_data_from_files(
        job_description_path=job_description_path,
        candidate_cv_path=candidate_cv_path,
        candidate_info_path=candidate_info_path,
        interview_transcript_path=interview_transcript_path
    )
    
    # Run evaluation
    return evaluate_interview_comprehensive(
        job_description=data["job_description"],
        candidate_cv=data["candidate_cv"],
        candidate_info=data["candidate_info"],
        interview_transcript=data["interview_transcript"],
        evaluation_instruction=evaluation_instruction,
        use_api=use_api
    )


def test_evaluation():
    """Test function to demonstrate how to use the evaluate_interview function"""

    # Load the JSON data from fi.json
    try:
        with open("fi.json", "r") as f:
            interview_data = json.load(f)

        # Evaluate the interview
        job_role = "AI Engineer"
        evaluation = evaluate_interview(job_role, interview_data)

        # Print the results
        print("=" * 60)
        print("INTERVIEW EVALUATION RESULTS")
        print("=" * 60)
        print(f"Job Role: {job_role}")
        print(f"Overall Score: {evaluation['overall_score']}/10")
        print(f"Hiring Recommendation: {evaluation['hiring_recommendation']}")
        print()

        print("CATEGORY SCORES:")
        print(f"  Technical Competency: {evaluation['technical_competency']}/10")
        print(f"  Problem Solving: {evaluation['problem_solving']}/10")
        print(f"  Communication: {evaluation['communication']}/10")
        print(f"  Experience Level: {evaluation['experience_level']}/10")
        print(f"  Cultural Fit: {evaluation['cultural_fit']}/10")
        print()

        print("STRENGTHS:")
        for strength in evaluation['strengths']:
            print(f"  • {strength}")
        print()

        print("AREAS FOR IMPROVEMENT:")
        for area in evaluation['areas_for_improvement']:
            print(f"  • {area}")
        print()

        print("DETAILED FEEDBACK:")
        print(evaluation['detailed_feedback'])
        print()

        print("INDIVIDUAL QUESTION SCORES:")
        print("-" * 40)
        for q_score in evaluation['individual_question_scores']:
            print(f"Question {q_score['question_number']}: {q_score['score']}/10")
            print(f"Q: {q_score['question_text']}")
            print(f"A: {q_score['answer_text']}")
            print(f"Feedback: {q_score['feedback']}")
            print("-" * 40)

        # Save the evaluation to a file
        with open("evaluation_result.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nEvaluation saved to evaluation_result.json")

    except FileNotFoundError:
        print("Error: fi.json file not found. Please make sure the file exists.")
    except Exception as e:
        print(f"Error: {e}")


def example_usage():
    """
    Example usage of comprehensive evaluation with the data structure provided.
    This demonstrates how to use the evaluation with job description, CV, and transcript.
    """
    # Example data structure matching the API endpoint format
    example_data = {
        "job_description": {
            "title": "Data Scientist",
            "department": "Technology & Innovation",
            "location": "Remote",
            "employment_type": "Full-time",
            "summary": "We are seeking a Data Scientist to build predictive models, analyze large datasets, and support data-driven decision-making within our product and marketing teams.",
            "responsibilities": [
                "Develop and deploy machine learning models for business insights.",
                "Clean, process, and analyze structured and unstructured data.",
                "Collaborate with engineers to integrate AI solutions into production.",
                "Communicate findings through reports and dashboards."
            ],
            "requirements": [
                "Bachelor's or Master's degree in Computer Science, Statistics, or related field.",
                "3+ years of experience in data science or machine learning roles.",
                "Proficiency in Python, SQL, and machine learning frameworks (scikit-learn, TensorFlow, PyTorch).",
                "Strong problem-solving and communication skills."
            ],
            "skills": ["Python", "Machine Learning", "SQL", "Data Visualization", "Statistics"]
        },
        "candidate_info": {
            "candidate_id": "cand_055",
            "candidate_email": "test55@gmail.com",
            "candidate_name": "John Best",
            "job_id": "job_55"
        },
        "candidate_cv": {
            "education": [
                {
                    "degree": "MSc. Data Science",
                    "institution": "University of Cape Town",
                    "year_completed": 2022
                },
                {
                    "degree": "BSc. Computer Science",
                    "institution": "Obafemi Awolowo University",
                    "year_completed": 2018
                }
            ],
            "experience": [
                {
                    "position": "Machine Learning Engineer",
                    "company": "TechNova Analytics",
                    "duration": "2022 - Present",
                    "achievements": [
                        "Developed customer churn prediction model with 89% accuracy.",
                        "Built recommendation engine improving sales conversion by 23%."
                    ]
                },
                {
                    "position": "Data Analyst",
                    "company": "Innova Solutions",
                    "duration": "2019 - 2022",
                    "achievements": [
                        "Automated reporting pipeline saving 10 hours per week.",
                        "Conducted A/B testing to optimize product pricing."
                    ]
                }
            ],
            "skills": ["Python", "SQL", "TensorFlow", "Power BI", "Data Visualization", "AWS"],
            "certifications": [
                "AWS Certified Machine Learning – Specialty (2023)",
                "Google Data Analytics Professional Certificate (2021)"
            ],
            "summary": "Data Scientist with 5 years of experience in building and deploying predictive models, data pipelines, and visualization dashboards to drive business intelligence and decision-making."
        },
        "interview_transcript": {
            "session_history": {
                "items": []
            },
            "real_time_transcript": {
                "user_transcript": "",
                "agent_transcript": "",
                "conversation_history": []
            }
        },
        "evaluation_instruction": "Evaluate candidate fitness for the Data Scientist role and provide a summary score based on technical skills, experience, and alignment with job requirements."
    }
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    evaluation = evaluate_interview_comprehensive(
        job_description=example_data["job_description"],
        candidate_cv=example_data["candidate_cv"],
        candidate_info=example_data["candidate_info"],
        interview_transcript=example_data["interview_transcript"],
        evaluation_instruction=example_data["evaluation_instruction"],
        use_api=False  # Set to True to use external API
    )
    
    # Save results
    with open("evaluation_result.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Candidate: {evaluation.get('candidate_name', 'N/A')}")
    print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
    print(f"Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A')}")
    print("\nEvaluation saved to evaluation_result.json")
    
    return evaluation


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If transcript path provided, use it
        from prompt import load_transcript_from_file
        
        transcript_path = sys.argv[1]
        
        # Example: Load transcript and run evaluation
        # You would need to provide job_description, candidate_cv, and candidate_info
        print(f"Loading transcript from: {transcript_path}")
        transcript = load_transcript_from_file(transcript_path)
        
        # For demo, use example data structure
        # In production, load these from files or API
        example_data = {
            "job_description": {
                "title": "Data Scientist",
                "department": "Technology & Innovation",
                "location": "Remote",
                "employment_type": "Full-time",
                "summary": "We are seeking a Data Scientist...",
                "responsibilities": [],
                "requirements": [],
                "skills": []
            },
            "candidate_info": {
                "candidate_id": "cand_001",
                "candidate_email": "candidate@example.com",
                "candidate_name": "Candidate Name",
                "job_id": "job_001"
            },
            "candidate_cv": {
                "education": [],
                "experience": [],
                "skills": [],
                "summary": ""
            }
        }
        
        evaluation = evaluate_interview_comprehensive(
            job_description=example_data["job_description"],
            candidate_cv=example_data["candidate_cv"],
            candidate_info=example_data["candidate_info"],
            interview_transcript=transcript,
            use_api=False
        )

        with open("evaluation_result.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"\nEvaluation saved to evaluation_result.json")
    else:
        # Run example usage
        example_usage()
