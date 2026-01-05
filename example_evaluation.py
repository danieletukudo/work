"""
Example script showing how to use the comprehensive evaluation system
with job description, candidate CV, and interview transcript.
"""

import json
from evaluate import evaluate_interview_comprehensive
from prompt import load_transcript_from_file

# Example data structure matching the API endpoint format
def run_evaluation_example():
    """Example of running evaluation with all required data"""
    
    # 1. Job Description
    job_description = {
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
    }
    
    # 2. Candidate Information
    candidate_info = {
        "candidate_id": "cand_055",
        "candidate_email": "test55@gmail.com",
        "candidate_name": "John Best",
        "job_id": "job_55"
    }
    
    # 3. Candidate CV
    candidate_cv = {
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
    }
    
    # 4. Load interview transcript from file
    transcript_path = "downloads/transcript_20251128_182926.json"
    try:
        interview_transcript = load_transcript_from_file(transcript_path)
        print(f" Loaded transcript from: {transcript_path}")
    except FileNotFoundError:
        print(f" Transcript file not found: {transcript_path}")
        print("Using empty transcript for example...")
        interview_transcript = {
            "session_history": {"items": []},
            "real_time_transcript": {
                "user_transcript": "",
                "agent_transcript": "",
                "conversation_history": []
            }
        }
    
    # 5. Evaluation instruction (optional)
    evaluation_instruction = "Evaluate candidate fitness for the Data Scientist role and provide a summary score based on technical skills, experience, and alignment with job requirements."
    
    # 6. Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 60)
    print(f"Candidate: {candidate_info['candidate_name']}")
    print(f"Job: {job_description['title']}")
    print(f"Job ID: {candidate_info['job_id']}")
    print("\nEvaluating...")
    
    evaluation = evaluate_interview_comprehensive(
        job_description=job_description,
        candidate_cv=candidate_cv,
        candidate_info=candidate_info,
        interview_transcript=interview_transcript,
        evaluation_instruction=evaluation_instruction,
        use_api=False  # Set to True to use external API endpoint
    )
    
    # 7. Save results
    output_file = "evaluation_result.json"
    with open(output_file, "w") as f:
        json.dump(evaluation, f, indent=2)
    
    # 8. Display summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
    print(f"Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A')}")
    print(f"\nCategory Scores:")
    print(f"  Technical Competency: {evaluation.get('technical_competency', 'N/A')}/10")
    print(f"  Problem Solving: {evaluation.get('problem_solving', 'N/A')}/10")
    print(f"  Communication: {evaluation.get('communication', 'N/A')}/10")
    print(f"  Experience Level: {evaluation.get('experience_level', 'N/A')}/10")
    print(f"  Cultural Fit: {evaluation.get('cultural_fit', 'N/A')}/10")
    print(f"  CV Alignment: {evaluation.get('cv_alignment', 'N/A')}/10")
    print(f"  Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10")
    
    if evaluation.get('strengths'):
        print(f"\nStrengths:")
        for strength in evaluation['strengths']:
            print(f"  • {strength}")
    
    if evaluation.get('areas_for_improvement'):
        print(f"\nAreas for Improvement:")
        for area in evaluation['areas_for_improvement']:
            print(f"  • {area}")
    
    print(f"\n Full evaluation saved to: {output_file}")
    print("=" * 60)
    
    return evaluation


if __name__ == "__main__":
    run_evaluation_example()

