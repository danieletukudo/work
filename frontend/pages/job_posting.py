import streamlit as st
import json
import numpy as np
import requests

# --- SMART SUGGESTION MAPPINGS --- #
JOB_SUGGESTIONS = {
    "Data Scientist": {
        "skills": ["Python", "Machine Learning", "SQL", "Data Visualization", "Pandas", "TensorFlow"],
        "functions": ["Model Development", "Data Analysis", "Feature Engineering"]
    },
    "Software Engineer": {
        "skills": ["JavaScript", "React", "Node.js", "Git", "API Development"],
        "functions": ["Frontend Development", "Backend Development", "System Design"]
    },
    "Product Manager": {
        "skills": ["Agile", "Roadmapping", "Stakeholder Management", "User Research"],
        "functions": ["Product Strategy", "Requirements Definition", "Team Coordination"]
    },
    "Data Analyst": {
        "skills": ["Excel", "SQL", "Power BI", "Data Cleaning", "Dashboarding"],
        "functions": ["Data Reporting", "Trend Analysis", "Business Insights"]
    },
    "UX Designer": {
        "skills": ["Figma", "Wireframing", "User Testing", "Prototyping"],
        "functions": ["Design Research", "User Flow Design", "Usability Testing"]
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "MLOps"],
        "functions": ["Model Development", "Algorithm Design", "Data Pipeline", "Research"]
    },
    "Software Developer": {
        "skills": ["JavaScript", "React", "Node.js", "Git", "API Development"],
        "functions": ["Frontend Development", "Backend Development", "System Design"]
    },
    "Marketing Specialist": {
        "skills": ["Digital Marketing", "SEO", "Content Creation", "Analytics", "Social Media"],
        "functions": ["Campaign Management", "Market Research", "Content Strategy"]
    },
    "Sales Executive": {
        "skills": ["Negotiation", "CRM", "Lead Generation", "Presentation", "Relationship Building"],
        "functions": ["Sales Strategy", "Client Acquisition", "Revenue Growth"]
    },
    "Financial Analyst": {
        "skills": ["Financial Modeling", "Excel", "Data Analysis", "Accounting", "Forecasting"],
        "functions": ["Financial Reporting", "Budget Analysis", "Investment Analysis"]
    },
    "DevOps Engineer": {
        "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux", "Infrastructure"],
        "functions": ["Infrastructure Management", "Deployment Automation", "System Monitoring"]
    }
}

API_ENDPOINT = "http://localhost:8000/generate-job-description"

def generate_job_description(payload):
    """Send job details to FastAPI backend and get AI-generated description."""
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("job_description", "")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return "Error: Failed to generate job description."
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return "Error: Could not connect to the backend."

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Job Posting Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Powered Job Post Creation 🤖")
st.markdown("---")

# ---------------- PRESETS ---------------- #
job_titles = [
    "Data Scientist", "Machine Learning Engineer", "Software Developer",
    "Product Manager", "Marketing Specialist", "Sales Executive",
    "Financial Analyst", "UX Designer", "DevOps Engineer"
]

industries = [
    "Technology", "Finance", "Healthcare", "Education",
    "Manufacturing", "Retail", "Energy", "Consulting", "Media & Communications"
]

job_functions = [
    "Engineering", "Marketing", "Sales", "Operations",
    "Customer Support", "Human Resources", "Design", "Data Analytics", "Management"
]

skills_list = [
    "Python", "SQL", "Data Analysis", "Communication",
    "Leadership", "Machine Learning", "Project Management", "React", "Cloud Computing"
]

timezones = [
    "Eastern Time Zone (ET)",
    "Central Time Zone (CT)",
    "Mountain Time Zone (MT)",
    "Pacific Time Zone (PT)"
]

# ---------------- SESSION STATE INITIALIZATION ---------------- #
if "job_id" not in st.session_state:
    st.session_state.job_id = "job_" + str(np.random.randint(1000, 9999))
if "generated_description" not in st.session_state:
    st.session_state.generated_description = ""
if "manual_description" not in st.session_state:
    st.session_state.manual_description = ""

# Initialize job title and suggestions
if "current_job_title" not in st.session_state:
    st.session_state.current_job_title = job_titles[0]
    # Initialize with default suggestions for the first job title
    if job_titles[0] in JOB_SUGGESTIONS:
        st.session_state.selected_skills = JOB_SUGGESTIONS[job_titles[0]]["skills"][:3]
        st.session_state.selected_functions = JOB_SUGGESTIONS[job_titles[0]]["functions"][:3]
    else:
        st.session_state.selected_skills = []
        st.session_state.selected_functions = []

# Function to update suggestions when job title changes
def update_suggestions(job_title):
    if job_title in JOB_SUGGESTIONS:
        st.session_state.selected_skills = JOB_SUGGESTIONS[job_title]["skills"][:3]
        st.session_state.selected_functions = JOB_SUGGESTIONS[job_title]["functions"][:3]
    else:
        st.session_state.selected_skills = []
        st.session_state.selected_functions = []
    st.session_state.current_job_title = job_title

# ---------------- JOB TITLE SELECTION (Outside Form) ---------------- #
st.header("1. Enter Job Details")

# Job title selection outside the form so we can use on_change
job_title = st.selectbox(
    "Job Title", 
    job_titles,
    key="job_title_select",
    index=job_titles.index(st.session_state.current_job_title) if st.session_state.current_job_title in job_titles else 0,
    on_change=lambda: update_suggestions(st.session_state.job_title_select)
)

# ---------------- FORM UI ---------------- #
with st.form(key="job_details_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # # Display current job title (read-only)
        # st.text_input("Selected Job Title", value=job_title, disabled=True)
        
        job_level = st.selectbox("Job Level", ["Entry-Level", "Mid-Level", "Senior", "Manager"])
        work_type = st.selectbox("Work Type", ["Full-Time", "Part-Time", "Contract", "Internship"])
        selected_industries = st.multiselect("Industry (Select one or more)", industries)
    
    with col2:
        # Get all available options
        all_skills = list(set(skills_list + [s for v in JOB_SUGGESTIONS.values() for s in v["skills"]]))
        all_functions = list(set(job_functions + [f for v in JOB_SUGGESTIONS.values() for f in v["functions"]]))
        
        # Multiselect with current session state values
        selected_skills = st.multiselect(
            "Key Skills",
            options=all_skills,
            default=st.session_state.selected_skills
        )

        selected_functions = st.multiselect(
            "Job Functions",
            options=all_functions,
            default=st.session_state.selected_functions
        )

        timezone = st.selectbox("Time Zone", timezones)

    other_details = st.text_area(
        "Additional Skills or Details",
        placeholder="""Are there other skills or details to include?
        (e.g., emphasize teamwork, innovation, company mission, or include benefits...)"""
    )

    generate_button = st.form_submit_button(label="Generate Job Description")

    # Update session state with current selections when form is submitted
    if generate_button:
        st.session_state.selected_skills = selected_skills
        st.session_state.selected_functions = selected_functions

## --- AI Description Generation --- #
if generate_button:
    if not job_title or not selected_skills:
        st.error("Please provide at least a Job Title and Key Skills.")
    else:
        payload = {
            "job_id": st.session_state.job_id,
            "job_title": job_title,
            "skills": selected_skills,
            "industries": selected_industries,
            "job_functions": selected_functions,
            "employer_id": "EMP-125",
            "job_level": job_level,
            "work_type": work_type,
            "timezone": timezone,
            "other_details": other_details
        }

        with st.spinner("Generating optimized job description..."):
            description = generate_job_description(payload)
            st.session_state.generated_description = description
            st.session_state.manual_description = description

        st.success("✅ Job description generated successfully!")

# --- Always Visible Job Description Box (Reactive) --- #
st.markdown("---")
st.header("2. Job Description")

# Use session state directly so update reflects immediately
manual_description = st.text_area(
    "Enter or Edit Job Description",
    # value=st.session_state.get("manual_description", ""),
    height=600,
    placeholder="Paste an existing job description here, or generate one above.",
    key="manual_description"
)

# --- Finalize Posting --- #
if st.button("Finalize & Post Job"):
    if not manual_description.strip():
        st.error("Please provide a job description before finalizing.")
    else:
        final_post = {
            "title": job_title,
            "description": manual_description.strip(),
            "metadata": {
                "job_id": st.session_state.job_id,
                "industries": selected_industries,
                "functions": selected_functions,
                "level": job_level,
                "work_type": work_type,
                "timezone": timezone,
                "skills": selected_skills,
            }
        }
        st.success("🎉 Job post finalized successfully!")
        st.json(final_post)

# Debugging
# with st.expander("Debug Info"):
#     st.write("Current Job Title:", st.session_state.current_job_title)
#     st.write("Selected Skills:", st.session_state.selected_skills)
#     st.write("Selected Functions:", st.session_state.selected_functions)
#     st.write("Form Job Title:", job_title)