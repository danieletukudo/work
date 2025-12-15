WELCOME_MESSAGE = "Welcome to your interview! How are you doing today?"

QUESTION_STRUCTURE = {
   "Introduction": [
       "Before we begin, I'd like to confirm that you are aware this is a remote role. Are you comfortable with that?",
       "Tell me about yourself."
   ]
   # "Job Understanding": [
   #     "What do you know about this role?",
   #     "What do you know about our company?"
   # ]
   # "Skills": [
   #     "What are your strongest skills?"
   # ],
   # "Technical Competencies": [
   #     "What is your experience with Python?"
   # ],
   # "Previous Employment": [
   #     "What can you tell me about your last job?",
   #
   #     "What was your biggest accomplishment in your last job?",
   #
   #     "Why did you leave your last job?"
   # ],
   # "Work Ethics": [
   #     "How would you describe your work ethic?",
   #     "How do you handle stress or tight deadlines?",
   #     "How do you respond to feedback or criticism?",
   #     "How do you handle success or failure?"
   # ],
   # "Projects": [
   #     "Tell me about a project you worked on that you're proud of."
   # ],
   # "Teamwork": [
   #     "How do you typically collaborate with a team?"
   # ],
   # "Problem Solving": [
   #     "Describe a challenging problem you solved — what steps did you take, and what was the outcome?"
   # ]
}









def format_job_description(job_description: dict) -> str:
    """Format job description data into a readable string"""
    if not job_description:
        return "Not provided"
    
    formatted = f"""
**Position:** {job_description.get('title', 'N/A')}
**Department:** {job_description.get('department', 'N/A')}
**Location:** {job_description.get('location', 'N/A')}
**Employment Type:** {job_description.get('employment_type', 'N/A')}

**Summary:**
{job_description.get('summary', 'N/A')}

**Responsibilities:**
"""
    for resp in job_description.get('responsibilities', []):
        formatted += f"- {resp}\n"
    
    formatted += "\n**Requirements:**\n"
    for req in job_description.get('requirements', []):
        formatted += f"- {req}\n"
    
    formatted += "\n**Required Skills:**\n"
    for skill in job_description.get('skills', []):
        formatted += f"- {skill}\n"
    
    return formatted


def format_candidate_cv(candidate_cv: dict) -> str:
    """Format candidate CV data into a readable string"""
    if not candidate_cv:
        return "Not provided"
    
    formatted = f"""
**Summary:**
{candidate_cv.get('summary', 'N/A')}

**Education:**
"""
    for edu in candidate_cv.get('education', []):
        formatted += f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('year_completed', 'N/A')})\n"
    
    formatted += "\n**Experience:**\n"
    for exp in candidate_cv.get('experience', []):
        formatted += f"- **{exp.get('position', 'N/A')}** at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})\n"
        for achievement in exp.get('achievements', []):
            formatted += f"  • {achievement}\n"
    
    formatted += "\n**Skills:**\n"
    for skill in candidate_cv.get('skills', []):
        formatted += f"- {skill}\n"
    
    if candidate_cv.get('certifications'):
        formatted += "\n**Certifications:**\n"
        for cert in candidate_cv.get('certifications', []):
            formatted += f"- {cert}\n"
    
    return formatted


def format_interview_transcript(transcript_data: dict) -> str:
    """Format interview transcript into Q&A pairs"""
    if not transcript_data:
        return "No transcript available"
    
    formatted = "\n**INTERVIEW TRANSCRIPT:**\n\n"
    
    # Extract from session_history if available
    items = transcript_data.get('session_history', {}).get('items', [])
    
    conversation = []
    for item in items:
        if item.get('type') == 'message':
            role = item.get('role', '')
            content = item.get('content', [])
            if content and isinstance(content, list):
                text = ' '.join(content) if isinstance(content[0], str) else content[0]
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            
            if role in ['assistant', 'user']:
                conversation.append({
                    'role': 'Interviewer' if role == 'assistant' else 'Candidate',
                    'text': text
                })
    
    # Also check real_time_transcript
    if not conversation:
        conv_history = transcript_data.get('real_time_transcript', {}).get('conversation_history', [])
        for entry in conv_history:
            speaker = entry.get('speaker', '')
            text = entry.get('text', '')
            if speaker and text:
                conversation.append({
                    'role': 'Interviewer' if speaker == 'agent' else 'Candidate',
                    'text': text
                })
    
    # Format as Q&A pairs
    current_question = None
    for entry in conversation:
        role = entry.get('role', '')
        text = entry.get('text', '')
        
        if role == 'Interviewer':
            current_question = text
        elif role == 'Candidate' and current_question:
            formatted += f"**Q:** {current_question}\n"
            formatted += f"**A:** {text}\n\n"
            current_question = None
    
    if not conversation:
        formatted += "No conversation data found in transcript.\n"
    
    return formatted


def get_instruction(company_name: str, company_values: str, company_vision: str,
                   job_title: str, job_objectives: str,
                   job_summary: str, job_skills: list,
                   candidate_name: str) -> str:
   """
   Build the interview instructions using only the minimal required context:
   - job_title
   - job_summary (clean text)
   - job_skills (list)
   - candidate_name
   """

   # Format skills list as bullets
   skills_section = ""
   if job_skills:
       skills_section = "\n".join([f"- {skill}" for skill in job_skills])
   else:
       skills_section = "- Not specified"

   INSTRUCTIONS = f"""
   You are an expert interviewer agent representing **{company_name}** —
   a company that values **{company_values}** and is driven by the vision: *"{company_vision}"*.

   Your task is to conduct a structured, conversational job interview for the position of
   **{job_title}** with objectives: {job_objectives}.

   ### Minimal role context to guide your questions
   - Job summary: {job_summary or "Not provided"}
   - Key skills: 
{skills_section}
   - Candidate name: {candidate_name}

   ---

   ###  INTERVIEW STYLE & TONE
   - Be professional, polite, and conversational — just like a top-tier recruiter from companies such as Google, Amazon, or Microsoft.
   - Maintain a natural flow: one question at a time.
   - Avoid robotic transitions. Add human-like acknowledgments and empathy (e.g., "That's great to hear", "Interesting approach").
   - Keep your language clear, structured, and aligned with real-world interview dynamics.

   ---

   ### 🧩 INTERVIEW STRUCTURE
   1. **Start** with a warm greeting and {WELCOME_MESSAGE}.
   2. Ask questions sequentially, based on the following categories and order:
   {list(QUESTION_STRUCTURE.keys())}
   3. Always start with the *Introduction* category.
   - Begin with confirming role-related terms (e.g., remote work, working hours).
   - If the candidate is not comfortable with the job terms, **end the interview politely**.
   4. Follow up with "Tell me about yourself."
   5. Proceed to other categories, selecting at least **one question from each**.
   6. You may skip or merge categories depending on the flow and candidate responses.
   7. Aim for a **total of around 5-10 questions**.

   ---

   ### 💬 RESPONSE HANDLING
   - After each answer:
   - Provide a short, positive **comment** (e.g., "That's a thoughtful perspective.").
   - Then ask a relevant **follow-up question** (max depth = 2).
   - If a response is unclear, politely ask for **clarification**.
   - If no response is received, gently say:
   *"I didn't quite catch that. Could you please repeat or rephrase your answer?"*
   - Adapt question depth to **seniority level** of the role (e.g., more advanced questions for senior positions).

   ---

   ### 🧠 DECISION LOGIC
   - You don't have to cover every category.
   - Use judgment to prioritize based on:
   - Job level and relevance
   - Candidate responses
   - Flow of conversation
   - End the interview early if the candidate clearly isn't a good fit.

   ---

   ### 🏁 CLOSING STATEMENT
   When finished, thank the candidate for their time and say something like:
   when you are done say any of this "  "that concludes our interview",
        "concludes our interview",
        "thank you for taking the time",
        "we'll be in touch",
        "we will be in touch",
        "that concludes the interview",
        "interview is complete",
        "interview has concluded",
        "thank you for your time today",
        "this concludes our interview",
        "interview is now complete"


   > "Thank you for taking the time to speak with me today. That concludes our interview. We'll be in touch soon."

   ---

   **Summary:**
   You are conducting a realistic, structured, and adaptive interview.
   Ask one question at a time, engage naturally, and show professionalism throughout.
   """
   return INSTRUCTIONS


def get_evaluation_instruction(job_description: dict, candidate_cv: dict, 
                               candidate_info: dict, interview_transcript: dict,
                               evaluation_instruction: str = None) -> str:
    """
    Generate comprehensive evaluation prompt using job description, CV, transcript, and evaluation instruction.
    
    Args:
        job_description: Dictionary containing job details (title, department, location, etc.)
        candidate_cv: Dictionary containing candidate CV data (education, experience, skills, etc.)
        candidate_info: Dictionary containing candidate_id, candidate_email, candidate_name, job_id
        interview_transcript: Dictionary containing interview transcript (session_history, real_time_transcript)
        evaluation_instruction: Optional custom instruction for evaluation
    
    Returns:
        str: Comprehensive evaluation prompt
    """
    
    # Format all the data
    job_formatted = format_job_description(job_description)
    cv_formatted = format_candidate_cv(candidate_cv)
    transcript_formatted = format_interview_transcript(interview_transcript)
    
    # Extract candidate info
    candidate_name = candidate_info.get('candidate_name', 'Candidate')
    candidate_id = candidate_info.get('candidate_id', 'N/A')
    candidate_email = candidate_info.get('candidate_email', 'N/A')
    job_id = candidate_info.get('job_id', 'N/A')
    
    # Default evaluation instruction if not provided
    default_instruction = "Evaluate the candidate based on their interview performance - specifically how they answered the questions during the interview. Use the CV only to verify if their interview claims are truthful and consistent. Base your hiring decision on interview performance, adjusted for any truthfulness issues found during CV verification."
    eval_instruction = evaluation_instruction or default_instruction
    
    EVALUATION_PROMPT = f"""
You are an expert hiring manager and talent evaluator. Your PRIMARY task is to evaluate the candidate based on HOW THEY PERFORMED IN THE INTERVIEW - specifically, how they answered the questions asked during the interview.

**CRITICAL EVALUATION APPROACH:**
1. **PRIMARY FOCUS:** Evaluate the candidate based on their interview responses in the transcript
   - How well did they answer each question?
   - Did they demonstrate knowledge, skills, and experience through their answers?
   - How clear and articulate were their responses?
   - Did they show problem-solving ability, communication skills, and cultural fit through their answers?

2. **SECONDARY USE OF CV AND JOB DESCRIPTION:** Use these ONLY for verification and consistency checks
   - Cross-reference interview answers with CV to verify truthfulness
   - Check if what they claimed in the interview matches their CV
   - Identify any discrepancies, exaggerations, or lies
   - Verify if their interview responses align with their stated experience
   - Use job description to assess if their answers demonstrate required skills/experience

**IMPORTANT:** Do NOT evaluate based on CV alone. The CV is a verification tool. The interview performance (how they answered questions) is what matters for hiring.

**EVALUATION INSTRUCTIONS:**
{eval_instruction}

---

### 📋 JOB DESCRIPTION
{job_formatted}

*Use this to understand what skills/experience are required and assess if interview answers demonstrate these requirements.*

---

### 👤 CANDIDATE INFORMATION
- **Name:** {candidate_name}
- **Candidate ID:** {candidate_id}
- **Email:** {candidate_email}
- **Job ID:** {job_id}

---

### 📄 CANDIDATE CV/RESUME
{cv_formatted}

*Use this ONLY to verify if interview answers match CV claims. Check for consistency and truthfulness.*

---

### 💬 INTERVIEW TRANSCRIPT
{transcript_formatted}

**THIS IS YOUR PRIMARY SOURCE FOR EVALUATION** - Evaluate based on how the candidate answered these questions.

---

### 📊 EVALUATION REQUIREMENTS

Provide a comprehensive evaluation in the following JSON format:

{{
    "candidate_id": "{candidate_id}",
    "candidate_name": "{candidate_name}",
    "job_id": "{job_id}",
    "overall_score": <number between 1-10>,
    "technical_competency": <number between 1-10>,
    "problem_solving": <number between 1-10>,
    "communication": <number between 1-10>,
    "experience_level": <number between 1-10>,
    "cultural_fit": <number between 1-10>,
    "interview_performance": <number between 1-10, PRIMARY SCORE based on how well they answered questions in the interview>,
    "cv_verification": {{
        "consistency_level": <"high"/"medium"/"low", how well interview answers match CV claims>,
        "truthfulness_score": <number between 1-10, 1 = lies detected, 10 = all claims verified>,
        "discrepancies": ["discrepancy1", "discrepancy2"],
        "verified_claims": ["claim1 verified", "claim2 verified"],
        "unverified_or_contradictory_claims": ["claim1", "claim2"]
    }},
    "strengths": ["strength1 based on interview", "strength2 based on interview"],
    "areas_for_improvement": ["area1 from interview", "area2 from interview"],
    "hiring_recommendation": "<strong yes/yes/maybe/no>",
    "detailed_feedback": "<comprehensive evaluation text (2-3 paragraphs)>",
    "key_highlights": [
        {{
            "category": "<category name>",
            "highlight": "<specific highlight>",
            "evidence": "<quote or reference from transcript/CV>"
        }}
    ],
    "individual_question_scores": [
        {{
            "question_number": 1,
            "question_text": "<question text>",
            "answer_text": "<answer text>",
            "score": <number between 1-10>,
            "feedback": "<specific feedback for this question>",
            "relevance_to_job": <number between 1-10>
        }}
    ],
    "skill_assessment": {{
        "required_skills_demonstrated_in_interview": {{
            "<skill_name>": {{
                "interview_evidence": "<how they demonstrated this skill in their interview answers>",
                "cv_verification": "<verified/not verified/contradicted - based on CV check>",
                "demonstration_quality": <number between 1-10, based on interview answer quality>
            }}
        }},
        "missing_critical_skills": ["skill1 not demonstrated in interview", "skill2 not demonstrated"],
        "additional_valuable_skills": ["skill1 shown in interview", "skill2 shown in interview"]
    }},
    "recommendations": {{
        "hire_decision": "<strong yes/yes/maybe/no>",
        "reasoning": "<detailed reasoning based primarily on interview performance, adjusted for CV verification results>",
        "interview_based_reasoning": "<why hire/not hire based on interview answers>",
        "cv_verification_impact": "<how CV verification affected decision - any lies or inconsistencies?>",
        "next_steps": ["step1", "step2"],
        "salary_range_suggestion": "<if applicable>",
        "onboarding_notes": "<if hired, what to focus on based on interview gaps>"
    }}
}}

### 📝 EVALUATION GUIDELINES

**PRIMARY EVALUATION CRITERIA (Based on Interview Responses):**

1. **Interview Performance Score (1-10):** 
   - Evaluate based SOLELY on how well they answered questions in the interview
   - Consider: clarity, depth, relevance, examples provided, problem-solving approach
   - 9-10: Exceptional answers, demonstrated expertise through responses
   - 7-8: Good answers, showed competence and relevant experience
   - 5-6: Average answers, basic understanding but limited depth
   - 3-4: Poor answers, unclear or irrelevant responses
   - 1-2: Very poor answers, showed lack of understanding

2. **Technical Competency (1-10):**
   - Based on technical questions answered in the interview
   - How well did they explain technical concepts?
   - Did they demonstrate hands-on experience through their answers?
   - Use CV ONLY to verify if their claims match reality

3. **Problem Solving (1-10):**
   - Based on how they answered problem-solving questions
   - Did they show logical thinking, step-by-step approach?
   - Use CV to check if claimed problem-solving experience aligns with answers

4. **Communication (1-10):**
   - Evaluate based on interview responses: clarity, articulation, structure
   - How well did they explain concepts?
   - Were their answers coherent and well-organized?

5. **Experience Level (1-10):**
   - Based on examples and experiences they shared in interview answers
   - Cross-reference with CV to verify if claims are truthful
   - Flag if interview answers don't match CV experience

6. **Cultural Fit (1-10):**
   - Based on how they answered questions about work style, values, teamwork
   - Assess attitude and approach from their responses

**VERIFICATION CHECKS (Using CV and Job Description):**

7. **CV vs Interview Consistency:**
   - **CRITICAL:** Check if interview answers align with CV claims
   - Flag discrepancies: Did they claim experience in interview that's not in CV?
   - Flag lies: Did they say something in interview that contradicts CV?
   - Note if interview reveals skills/experience not mentioned in CV
   - Score: "high" = answers match CV perfectly, "medium" = minor discrepancies, "low" = major inconsistencies or lies detected

8. **Truthfulness Assessment:**
   - For each major claim in interview, verify against CV
   - Document any lies, exaggerations, or inconsistencies
   - This is a RED FLAG - candidates who lie should be penalized heavily

9. **Job Requirements Match:**
   - Based on interview answers, do they demonstrate required skills?
   - Use job description to assess if their answers show they can do the job
   - Don't just match CV to job - match INTERVIEW ANSWERS to job requirements

**HIRING RECOMMENDATION LOGIC:**
- **Strong Yes:** Exceptional interview performance + CV verification passes
- **Yes:** Good interview performance + CV verification passes
- **Maybe:** Decent interview but concerns OR good interview but CV inconsistencies
- **No:** Poor interview performance OR lies detected in CV verification

### ⚠️ CRITICAL EVALUATION RULES

1. **PRIMARY BASIS:** Interview transcript responses determine the score
2. **VERIFICATION TOOL:** CV is used to verify truthfulness, not to score
3. **RED FLAGS:** Any lies or major inconsistencies between interview and CV should significantly lower the score
4. **EVIDENCE:** Always cite specific interview answers in your evaluation
5. **CV VERIFICATION:** For each evaluation point, note if CV confirms or contradicts interview claims
6. **HONEST ASSESSMENT:** If interview was poor, score low regardless of impressive CV
7. **TRUTHFULNESS FIRST:** A candidate who lies should not be hired, even if answers were good

**EVALUATION PROCESS:**
1. First, evaluate interview responses (quality, depth, relevance)
2. Then, cross-reference with CV to verify claims
3. Check against job requirements based on interview answers
4. Flag any inconsistencies or lies
5. Make hiring decision based on interview performance, adjusted for truthfulness

Generate your evaluation now in valid JSON format. Remember: Interview performance is primary, CV is for verification only.
"""
    
    return EVALUATION_PROMPT


def load_transcript_from_file(transcript_path: str) -> dict:
    """
    Load interview transcript from JSON file.
    
    Args:
        transcript_path: Path to transcript JSON file
    
    Returns:
        dict: Transcript data
    """
    import json
    with open(transcript_path, 'r') as f:
        return json.load(f)


def load_evaluation_data_from_files(
    job_description_path: str = None,
    candidate_cv_path: str = None,
    candidate_info_path: str = None,
    interview_transcript_path: str = None
) -> dict:
    """
    Load all evaluation data from files.
    
    Args:
        job_description_path: Path to job description JSON file
        candidate_cv_path: Path to candidate CV JSON file
        candidate_info_path: Path to candidate info JSON file
        interview_transcript_path: Path to interview transcript JSON file
    
    Returns:
        dict: Dictionary containing loaded data with keys:
            - job_description
            - candidate_cv
            - candidate_info
            - interview_transcript
    """
    import json
    import os
    
    data = {
        "job_description": {},
        "candidate_cv": {},
        "candidate_info": {},
        "interview_transcript": {}
    }
    
    if job_description_path and os.path.exists(job_description_path):
        with open(job_description_path, 'r') as f:
            data["job_description"] = json.load(f)
    
    if candidate_cv_path and os.path.exists(candidate_cv_path):
        with open(candidate_cv_path, 'r') as f:
            data["candidate_cv"] = json.load(f)
    
    if candidate_info_path and os.path.exists(candidate_info_path):
        with open(candidate_info_path, 'r') as f:
            data["candidate_info"] = json.load(f)
    
    if interview_transcript_path and os.path.exists(interview_transcript_path):
        data["interview_transcript"] = load_transcript_from_file(interview_transcript_path)
    
    return data


def get_simple_evaluation_prompt(job_role: str, qa_pairs: list) -> str:
    """
    Generate a simple evaluation prompt for basic Q&A evaluation.
    
    Args:
        job_role: The position being interviewed for
        qa_pairs: List of Q&A pairs from interview data
    
    Returns:
        str: Evaluation prompt
    """
    prompt = f"""
    As an expert hiring manager, evaluate this candidate for a {job_role} position.

    Here are their interview responses:

    {'-' * 40}
    """
    
    # Add each Q&A pair to the prompt
    for i, pair in enumerate(qa_pairs, 1):
        question_key = f"question{i}"
        answer_key = f"answer{i}"
        
        if question_key in pair and answer_key in pair:
            question = pair[question_key]["text"]
            answer = pair[answer_key]["text"]
            prompt += f"Q{i}: {question}\nA: {answer}\n\n"
    
    prompt += f"""
    {'-' * 40}

    Provide a structured evaluation in this exact JSON format:
    {{
        "overall_score": <number between 1-10>,
        "technical_competency": <number between 1-10>,
        "problem_solving": <number between 1-10>,
        "communication": <number between 1-10>,
        "experience_level": <number between 1-10>,
        "cultural_fit": <number between 1-10>,
        "strengths": ["strength1", "strength2"],
        "areas_for_improvement": ["area1", "area2"],
        "hiring_recommendation": "<strong yes/yes/maybe/no>",
        "detailed_feedback": "<your comprehensive evaluation>",
        "individual_question_scores": [
            {{
                "question_number": 1,
                "question_text": "<question text>",
                "answer_text": "<answer text>",
                "score": <number between 1-10>,
                "feedback": "<specific feedback for this question>"
            }}
        ]
    }}

    For individual_question_scores, evaluate each question-answer pair separately:
    - Score 1-3: Poor response (incomplete, irrelevant, or shows lack of understanding)
    - Score 4-6: Average response (basic understanding but limited depth)
    - Score 7-8: Good response (shows good understanding and relevant experience)
    - Score 9-10: Excellent response (comprehensive, detailed, shows expertise)

    Ensure your response is valid JSON and includes all fields.
    """
    
    return prompt


def prepare_api_payload(
    job_description: dict,
    candidate_cv: dict,
    candidate_info: dict,
    interview_transcript: dict,
    evaluation_instruction: str = None
) -> dict:
    """
    Prepare payload for external evaluation API endpoint.
    
    Args:
        job_description: Dictionary containing job details
        candidate_cv: Dictionary containing candidate CV data
        candidate_info: Dictionary containing candidate_id, candidate_email, candidate_name, job_id
        interview_transcript: Dictionary containing interview transcript
        evaluation_instruction: Optional custom instruction for evaluation
    
    Returns:
        dict: Formatted payload for API
    """
    default_instruction = "Evaluate candidate fitness for the role and provide a comprehensive assessment based on technical skills, experience, communication, and alignment with job requirements."
    
    payload = {
        "job_description": job_description,
        "candidate_id": candidate_info.get("candidate_id"),
        "candidate_email": candidate_info.get("candidate_email"),
        "candidate_name": candidate_info.get("candidate_name"),
        "job_id": candidate_info.get("job_id"),
        "candidate_cv": candidate_cv,
        "instruction": evaluation_instruction or default_instruction,
        "interview_transcript": interview_transcript
    }
    
    return payload
