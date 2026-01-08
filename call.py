from unittest import result

import requests



data={
        'status': True, 'message': 'CV data retrieved successfully',
        'data':
            {'job_description':
                 {'title': 'Customer Service Representative', 'department': 173, 'location': '', 'employment_type': 'Full time',
                  'summary': '"<h1>Job Overview</h1><p>We are seeking a highly motivated and results-oriented <strong>Customer'
                  ' Service Representative</strong> to provide exceptional customer support and drive customer satisfaction. '
                'This intermediate-level role is a full-time position focused on customer success and support.</p><p></p><h2>Ideal '
                'Candidate</h2><p>You are a proactive and customer-focused individual with a passion for helping customers succeed. You '
                 'possess excellent communication and problem-solving skills, and you are adept at identifying and capitalizing on market '
                'opportunities.</p><p></p><h2>Key Responsibilities</h2><h3>Customer Support</h3><ul><li>Provide timely and accurate responses to customer '
            'inquiries via phone, email, and chat.</li><li>Troubleshoot customer issues and escalate complex problems to the appropriate teams.</li><li>Maintain a'
            ' thorough understanding of our product offerings and effectively communicate their value to customers.</li></ul><h3>Sales and Business '
            'Development</h3><ul><li>Identify and pursue sales opportunities within the existing customer base.</li><li>Develop and maintain strong relationships with key'
            ' customers.</li><li>Contribute to business development efforts by identifying new market opportunities.</li></ul><h3>Management and Strategy</h3><ul><li>Assist in'
            ' the developmentv and implementation of customer service strategies.</li><li>Monitor customer feedback and identify areas for improvement.</li><li>Manage customer '
            'accounts and ensure customer satisfaction.</li></ul><h3>Product Knowledge and Market Awareness</h3><ul><li>Stay up-to-date on product updates and enhancements.</li><li>Monitor '
            'market trends and competitor activities.</li><li>Provide feedback to product development teams based on customer interactions.</li></ul><p></p><h2>Skill Sets Required / Preferred</h2><h3>Technical Skills</h3><ul><li>CRM software proficiency (e.g., Salesforce, Zendesk)</li><li>Proficiency in Microsoft Office Suite (Word, Excel, PowerPoint)</li><li>Data analysis and reporting skills</li></ul><h3>Soft Skills</h3><ul><li>Excellent  communication and interpersonal skills</li><li>Strong problem-solving and analytical abilities</li><li>Ability to work independently and as part of a team</li><li>Customer-focused and results-oriented</li><li>Strong organizational and time-management skills</li></ul><p></p><h2>Experience &amp; Education</h2><ul><li>Bachelors degree in a related field preferred</li><li>2+ years of experience in customer service, sales, or a related role</li><li>Proven track record of exceeding customer expectations</li><li>Experience with business development and market analysis is a plus</li></ul>"', 'responsibilities': '',
                  'requirements': [], 'skills':
                      ['Sales**r.w**Management**r.w**Market Opportunities**r.w**Product Offerings**r.w**Business Development']},


             'candidate_id':  '211',

             'candidate_email': 'testcan8@gmail.com',
             'candidate_name': 'Max Well',
             'job_id': '559',
             'job_application_id': '211',
             'candidate_cv': {'education': [],
                              'experience': [],
                              'skills': [],
                              'certifications': [],
                              'career_summary': 'candidate/570/resume/693b738f178d5_1765503887.pdf',}},}

def get_ai_cv_data(jobid, appid ):
    result = data
    import json
    import ast
    import json


    result = fetch_job_and_candidate_data(result)








    return result

import re
def clean_html(text):
    """Remove HTML tags and clean up text"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return text.replace('\\"', '').replace('"', '').strip()


def get_ai_cv_data(jobid, appid):
    """
    STRICT FIX: This function MUST return the dictionary.
    In your original code, it was returning the result of another function
    which was returning None, causing your crash.
    """
    return data


def fetch_job_and_candidate_data(jobid: str = None, appid: str = None):
    try:
        # Fetch the dict
        api_response = get_ai_cv_data(jobid, appid)

        # SAFETY CHECK: If api_response is None, stop here
        if api_response is None:
            print("Error: API response was None")
            return None

        # Use .get(key, {}) to ensure we always have a dict to call .get() on next
        inner_data = api_response.get('data', {})
        job_desc = inner_data.get('job_description', {})
        cand_cv = inner_data.get('candidate_cv', {})

        # Clean summary
        clean_summary = clean_html(job_desc.get('summary', ''))

        # Extract and format data
        extracted_data = {
            "status": api_response.get("status"),
            "message": api_response.get("message"),
            "job_title": job_desc.get("title"),
            "job_summary": clean_summary,
            "job_skills": job_desc.get("skills"),
            "candidate_name": inner_data.get("candidate_name"),
            "candidate_email": inner_data.get("candidate_email"),
            "cv_skills": cand_cv.get("skills"),
            "cv_file": cand_cv.get("career_summary")
        }

        return extracted_data

    except Exception as e:
        print(f"Error encountered: {e}")
        return None


if __name__ == "__main__":
    # Now this will run without the 'NoneType' error
    final_result = fetch_job_and_candidate_data(jobid="559", appid="211")
    print(final_result)