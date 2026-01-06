import re

def clean_html(raw_html):
    """Remove HTML tags from a string."""
    if not isinstance(raw_html, str):
        return raw_html
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    # Also handle the escaped quotes and newlines if present
    cleantext = cleantext.replace('\\"', '"').replace('\\n', '\n')
    return cleantext.strip()

# The data dictionary provided directly
data_dict = {'status': True, 'message': 'CV data retrieved successfully', 'data': {'job_description': {'title': '.Net Developer II', 'department': 96, 'location': '', 'employment_type': 'Full time', 'summary': '"<p style=\\"\\">We are seeking a highly skilled and experienced Full Stack Developer with a strong background in PHP (Laravel) and JavaScript (React.js) to join our development team. In this role, you will be responsible for designing, developing, and maintaining high-quality web applications that meet our clients\' needs and contribute to the success of our projects.</p><p style=\\"\\">Key Responsibilities:</p><p style=\\"\\">- Develop, test, and maintain robust, scalable, and high-performance web applications using PHP (Laravel) and JavaScript (React.js).</p><p style=\\"\\">- Collaborate with cross-functional teams, including designers, product managers, and other developers, to create seamless and engaging user experiences.</p><p style=\\"\\">- Write clean, maintainable, and efficient code following best practices and industry standards.</p><p style=\\"\\">- Troubleshoot, debug, and optimize existing code to ensure high availability and performance.</p><p style=\\"\\">- Participate in code reviews, providing constructive feedback to peers to maintain code quality.</p><p style=\\"\\">- Stay up-to-date with emerging technologies and industry trends to continuously improve skills and knowledge.</p><p style=\\"\\">- Assist in the development and implementation of APIs and third-party integrations.</p><p style=\\"\\">- Contribute to the architecture and design of new features and applications.</p><p style=\\"\\">Qualifications:</p><p style=\\"\\">- Bachelor\'s degree in Computer Science, Information Technology, or a related field (or equivalent work experience).</p><p style=\\"\\">- Proven experience as a Full Stack Developer, with a strong portfolio of web applications.</p><p style=\\"\\">- Proficiency in PHP and the Laravel framework.</p><p style=\\"\\">- Strong knowledge of JavaScript and experience with React.js.</p><p style=\\"\\">- Familiarity with front-end technologies such as HTML5, CSS3, and responsive design principles.</p><p style=\\"\\">- Experience with version control systems, preferably Git.</p><p style=\\"\\">- Understanding of RESTful APIs and asynchronous request handling.</p><p style=\\"\\">- Excellent problem-solving skills and attention to detail.</p><p style=\\"\\">- Strong communication and collaboration skills</p><p style=\\"\\">- Ability to work independently and as part of a team in a fast-paced environment.</p><p style=\\"\\">- Understanding of CI/CD processes is essential.</p><p style=\\"\\">Preferred Qualifications:</p><p style=\\"\\">- Experience with other JavaScript frameworks or libraries (e.g., Vue.js, Angular).</p><p style=\\"\\">- Knowledge of database management systems, particularly MySQL or PostgreSQL.</p><p style=\\"\\">- Experience with cloud platforms such as Azure is an added advantage.</p><p style=\\"\\">- Familiarity with containerization and orchestration tools (e.g., Docker, Kubernetes) is an added advantage.</p><p style=\\"\\\">Benefits:</p><p style=\\"\\">- Competitive salary and performance-based bonuses.</p><p style=\\"\\">- Flexible working hours and remote work options.</p><p style=\\"\\">- Comprehensive health, dental, and vision insurance.</p><p style=\\"\\">- Professional development opportunities and continuous learning.</p><p style=\\"\\">- Friendly and collaborative work environment.</p><p style=\\"\\">- Opportunities to work on exciting and challenging projects.</p>"', 'responsibilities': '', 'requirements': [], 'skills': ['Software Development and Programming**r.w**Marketing/ Social Media Assistant']}, 'candidate_id': 201, 'candidate_email': 'testcan4@gmail.com', 'candidate_name': 'Max Well', 'job_id': 164, 'candidate_cv': {'education': [], 'experience': [], 'skills': ['Customer Service', 'Marketing/ Social Media Assistant', ' Administrative Assistant', ' Executive Assistant'], 'certifications': [], 'career_summary': 'candidate/546/resume/68def49f38383_1759442079.pdf'}}}

def extract_info():
    try:
        # Use the dictionary directly
        parsed_data = data_dict
        
        print("=" * 60)
        print("EXTRACTED DATA REPORT")
        print("=" * 60)

        # 1. Top Level Status
        status = parsed_data.get('status')
        message = parsed_data.get('message')
        print(f"Status:  {'Success' if status else 'Failure'}")
        print(f"Message: {message}\n")

        if not status:
            return

        data_obj = parsed_data.get('data', {})
        
        # 2. Candidate Details
        print(f"--- CANDIDATE DETAILS ---")
        print(f"Name:  {data_obj.get('candidate_name', 'N/A')}")
        print(f"ID:    {data_obj.get('candidate_id', 'N/A')}")
        print(f"Email: {data_obj.get('candidate_email', 'N/A')}")
        print("")

        # 3. Job Details
        job_desc = data_obj.get('job_description', {})
        print(f"--- JOB DETAILS (ID: {data_obj.get('job_id', 'N/A')}) ---")
        print(f"Title:           {job_desc.get('title', 'N/A')}")
        print(f"Department:      {job_desc.get('department', 'N/A')}")
        print(f"Location:        {job_desc.get('location') or 'Not specified'}")



        print(f"Employment Type: {job_desc.get('employment_type', 'N/A')}")

        
        # Handling the HTML summary
        raw_summary = job_desc.get('summary', '')
        # The summary string in the input has an extra pair of double quotes at start/end: '"<p>..."'
        if raw_summary.startswith('"') and raw_summary.endswith('"'):
            raw_summary = raw_summary[1:-1]
        
        cleaned_summary = clean_html(raw_summary)
        # Indent the summary for better readability
        formatted_summary = "\n".join(["    " + line for line in cleaned_summary.split('\n') if line.strip()])
        
        print(f"Summary:\n{formatted_summary}")
        
        reqs = job_desc.get('requirements', [])
        print(f"Requirements:    {', '.join(reqs) if reqs else 'None listed'}")
        
        job_skills = job_desc.get('skills', [])
        print(f"Required Skills: {', '.join(job_skills) if job_skills else 'None listed'}")
        print("")

        # 4. Candidate CV Analysis
        cv_data = data_obj.get('candidate_cv', {})
        print(f"--- CANDIDATE CV ANALYSIS ---")
        
        cv_skills = cv_data.get('skills', [])
        print(f"Detected Skills: {', '.join(cv_skills) if cv_skills else 'None detected'}")
        
        education = cv_data.get('education', [])
        print(f"Education:       {education if education else 'None listed'}")
        
        experience = cv_data.get('experience', [])
        print(f"Experience:      {experience if experience else 'None listed'}")
        
        print(f"Resume File:     {cv_data.get('career_summary', 'N/A')}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    extract_info()
