"""
Utility functions to format evaluation results in different formats (JSON, TXT)
"""

import json
from typing import Dict
from datetime import datetime


def format_evaluation_as_txt(evaluation: Dict) -> str:
    """
    Convert evaluation JSON to a human-readable TXT format.
    
    Args:
        evaluation: Evaluation dictionary from evaluate_interview_comprehensive
    
    Returns:
        str: Formatted text representation of the evaluation
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("INTERVIEW EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Candidate Information
    lines.append("CANDIDATE INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Name: {evaluation.get('candidate_name', 'N/A')}")
    lines.append(f"Candidate ID: {evaluation.get('candidate_id', 'N/A')}")
    lines.append(f"Job ID: {evaluation.get('job_id', 'N/A')}")
    lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall Scores
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 80)
    lines.append(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
    lines.append(f"Interview Performance: {evaluation.get('interview_performance', 'N/A')}/10")
    lines.append(f"Hiring Recommendation: {evaluation.get('hiring_recommendation', 'N/A').upper()}")
    lines.append("")
    
    # Category Scores
    lines.append("DETAILED SCORES")
    lines.append("-" * 80)
    lines.append(f"Technical Competency: {evaluation.get('technical_competency', 'N/A')}/10")
    lines.append(f"Problem Solving: {evaluation.get('problem_solving', 'N/A')}/10")
    lines.append(f"Communication: {evaluation.get('communication', 'N/A')}/10")
    lines.append(f"Experience Level: {evaluation.get('experience_level', 'N/A')}/10")
    lines.append(f"Cultural Fit: {evaluation.get('cultural_fit', 'N/A')}/10")
    
    if 'cv_alignment' in evaluation:
        lines.append(f"CV Alignment: {evaluation.get('cv_alignment', 'N/A')}/10")
    lines.append("")
    
    # CV Verification
    cv_verification = evaluation.get('cv_verification', {})
    if cv_verification:
        lines.append("CV VERIFICATION")
        lines.append("-" * 80)
        lines.append(f"Consistency Level: {cv_verification.get('consistency_level', 'N/A').upper()}")
        lines.append(f"Truthfulness Score: {cv_verification.get('truthfulness_score', 'N/A')}/10")
        
        discrepancies = cv_verification.get('discrepancies', [])
        if discrepancies:
            lines.append("")
            lines.append("Discrepancies Found:")
            for i, disc in enumerate(discrepancies, 1):
                lines.append(f"  {i}. {disc}")
        
        verified = cv_verification.get('verified_claims', [])
        if verified:
            lines.append("")
            lines.append("Verified Claims:")
            for i, claim in enumerate(verified, 1):
                lines.append(f"  {i}. {claim}")
        
        unverified = cv_verification.get('unverified_or_contradictory_claims', [])
        if unverified:
            lines.append("")
            lines.append("Unverified/Contradictory Claims:")
            for i, claim in enumerate(unverified, 1):
                lines.append(f"  {i}. {claim}")
        lines.append("")
    
    # Strengths
    strengths = evaluation.get('strengths', [])
    if strengths:
        lines.append("STRENGTHS")
        lines.append("-" * 80)
        for i, strength in enumerate(strengths, 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
    
    # Areas for Improvement
    improvements = evaluation.get('areas_for_improvement', [])
    if improvements:
        lines.append("AREAS FOR IMPROVEMENT")
        lines.append("-" * 80)
        for i, area in enumerate(improvements, 1):
            lines.append(f"{i}. {area}")
        lines.append("")
    
    # Detailed Feedback
    detailed_feedback = evaluation.get('detailed_feedback', '')
    if detailed_feedback:
        lines.append("DETAILED FEEDBACK")
        lines.append("-" * 80)
        # Wrap long text
        words = detailed_feedback.split()
        current_line = []
        for word in words:
            if len(' '.join(current_line + [word])) > 75:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        lines.append("")
    
    # Individual Question Scores
    question_scores = evaluation.get('individual_question_scores', [])
    if question_scores:
        lines.append("QUESTION-BY-QUESTION ANALYSIS")
        lines.append("-" * 80)
        for q_score in question_scores:
            lines.append(f"Question {q_score.get('question_number', 'N/A')}: {q_score.get('score', 'N/A')}/10")
            lines.append(f"  Q: {q_score.get('question_text', 'N/A')}")
            lines.append(f"  A: {q_score.get('answer_text', 'N/A')}")
            lines.append(f"  Feedback: {q_score.get('feedback', 'N/A')}")
            if 'relevance_to_job' in q_score:
                lines.append(f"  Relevance to Job: {q_score.get('relevance_to_job', 'N/A')}/10")
            lines.append("")
    
    # Skill Assessment
    skill_assessment = evaluation.get('skill_assessment', {})
    if skill_assessment:
        lines.append("SKILL ASSESSMENT")
        lines.append("-" * 80)
        
        required_skills = skill_assessment.get('required_skills_demonstrated_in_interview', {})
        if required_skills:
            lines.append("Required Skills Demonstrated:")
            for skill_name, skill_data in required_skills.items():
                lines.append(f"  {skill_name}:")
                lines.append(f"    Demonstration Quality: {skill_data.get('demonstration_quality', 'N/A')}/10")
                lines.append(f"    Interview Evidence: {skill_data.get('interview_evidence', 'N/A')}")
                lines.append(f"    CV Verification: {skill_data.get('cv_verification', 'N/A')}")
            lines.append("")
        
        missing = skill_assessment.get('missing_critical_skills', [])
        if missing:
            lines.append("Missing Critical Skills:")
            for skill in missing:
                lines.append(f"  - {skill}")
            lines.append("")
        
        additional = skill_assessment.get('additional_valuable_skills', [])
        if additional:
            lines.append("Additional Valuable Skills:")
            for skill in additional:
                lines.append(f"  - {skill}")
            lines.append("")
    
    # Recommendations
    recommendations = evaluation.get('recommendations', {})
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append(f"Hire Decision: {recommendations.get('hire_decision', 'N/A').upper()}")
        lines.append("")
        
        reasoning = recommendations.get('reasoning', '')
        if reasoning:
            lines.append("Reasoning:")
            words = reasoning.split()
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) > 75:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            lines.append("")
        
        interview_reasoning = recommendations.get('interview_based_reasoning', '')
        if interview_reasoning:
            lines.append("Interview-Based Reasoning:")
            words = interview_reasoning.split()
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) > 75:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            lines.append("")
        
        cv_impact = recommendations.get('cv_verification_impact', '')
        if cv_impact:
            lines.append("CV Verification Impact:")
            words = cv_impact.split()
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) > 75:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            lines.append("")
        
        next_steps = recommendations.get('next_steps', [])
        if next_steps:
            lines.append("Next Steps:")
            for i, step in enumerate(next_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")
        
        onboarding = recommendations.get('onboarding_notes', '')
        if onboarding and onboarding != 'N/A':
            lines.append("Onboarding Notes:")
            words = onboarding.split()
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) > 75:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def save_evaluation_txt(evaluation: Dict, output_path: str) -> bool:
    """
    Save evaluation as TXT file.
    
    Args:
        evaluation: Evaluation dictionary
        output_path: Path to save the TXT file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        txt_content = format_evaluation_as_txt(evaluation)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        return True
    except Exception as e:
        print(f"Error saving TXT file: {e}")
        return False

