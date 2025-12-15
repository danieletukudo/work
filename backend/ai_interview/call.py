import requests

def get_ai_cv_data(jobid, appid):
        url = "https://bdev.remoting.work/api/v1/jobdescription/ai-cv-data"
        params = {
            "jobid": jobid,
            "appid": appid
        }

         # try:
        # response = requests.get(url, params=params, timeout=10)
        # response.raise_for_status()   # Will raise HTTPError if status is 4xx or 5xx
        data = {'status': True, 'message': 'CV data retrieved successfully', 'data': {'job_description': {'title': 'Customer Service Representative', 'department': 173, 'location': '', 'employment_type': 'Full time', 'summary': '"<h1>Job Overview</h1><p>We are seeking a highly motivated and results-oriented <strong>Customer Service Representative</strong> to provide exceptional customer support and drive customer satisfaction. This intermediate-level role is a full-time position focused on customer success and support.</p><p></p><h2>Ideal Candidate</h2><p>You are a proactive and customer-focused individual with a passion for helping customers succeed. You possess excellent communication and problem-solving skills, and you are adept at identifying and capitalizing on market opportunities.</p><p></p><h2>Key Responsibilities</h2><h3>Customer Support</h3><ul><li>Provide timely and accurate responses to customer inquiries via phone, email, and chat.</li><li>Troubleshoot customer issues and escalate complex problems to the appropriate teams.</li><li>Maintain a thorough understanding of our product offerings and effectively communicate their value to customers.</li></ul><h3>Sales and Business Development</h3><ul><li>Identify and pursue sales opportunities within the existing customer base.</li><li>Develop and maintain strong relationships with key customers.</li><li>Contribute to business development efforts by identifying new market opportunities.</li></ul><h3>Management and Strategy</h3><ul><li>Assist in the development and implementation of customer service strategies.</li><li>Monitor customer feedback and identify areas for improvement.</li><li>Manage customer accounts and ensure customer satisfaction.</li></ul><h3>Product Knowledge and Market Awareness</h3><ul><li>Stay up-to-date on product updates and enhancements.</li><li>Monitor market trends and competitor activities.</li><li>Provide feedback to product development teams based on customer interactions.</li></ul><p></p><h2>Skill Sets Required / Preferred</h2><h3>Technical Skills</h3><ul><li>CRM software proficiency (e.g., Salesforce, Zendesk)</li><li>Proficiency in Microsoft Office Suite (Word, Excel, PowerPoint)</li><li>Data analysis and reporting skills</li></ul><h3>Soft Skills</h3><ul><li>Excellent communication and interpersonal skills</li><li>Strong problem-solving and analytical abilities</li><li>Ability to work independently and as part of a team</li><li>Customer-focused and results-oriented</li><li>Strong organizational and time-management skills</li></ul><p></p><h2>Experience &amp; Education</h2><ul><li>Bachelor\'s degree in a related field preferred</li><li>2+ years of experience in customer service, sales, or a related role</li><li>Proven track record of exceeding customer expectations</li><li>Experience with business development and market analysis is a plus</li></ul>"', 'responsibilities': '', 'requirements': [], 'skills': ['Sales**r.w**Management**r.w**Market Opportunities**r.w**Product Offerings**r.w**Business Development']}, 'candidate_id': 211, 'candidate_email': 'testcan8@gmail.com', 'candidate_name': 'Max Well', 'job_id': 559, 'candidate_cv': {'education': [], 'experience': [], 'skills': [], 'certifications': [], 'career_summary': 'candidate/570/resume/693b738f178d5_1765503887.pdf'}}}

        return data

if __name__ == "__main__":
    data = get_ai_cv_data(jobid=164, appid=546)
    print("Response data:")
    print(data)
