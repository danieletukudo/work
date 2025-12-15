import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

# App configuration
st.set_page_config(page_title="CV Analyst", page_icon="🧠", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .title {
        font-size: 2.2em;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2em;
        font-weight: 500;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📄 CV Analyst Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze candidate CVs for a given job</div>', unsafe_allow_html=True)

headers = {
    "User-Agent": os.getenv("USER_AGENT"),
    "Referer": os.getenv("REFERER"),
    "Accept": os.getenv("ACCEPT"),
    "Accept-Language": os.getenv("ACCEPT_LANGUAGE"),
    "Connection": "keep-alive"
}

# Step 1: Input job ID

job_id = st.text_input("Enter Job ID:", placeholder="e.g., 220")
if st.button("Search Job"):
    if job_id.isdigit():
        try:
            response = requests.get(f"{BACKEND_URL}/job-details/{job_id}",
                                    headers=headers, allow_redirects=True)
            if response.status_code == 200:
                job_info = response.json()
                if job_info == "Job listing not found":
                    st.error("Job listing not found.")
                else:
                    st.session_state.job_info = job_info
                    st.success("Job Found ✅")
            else:
                st.error("Job not found or server error.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.warning("Please enter a valid numeric job ID.")

# Step 2: Display job details and upload additional CVs
if "job_info" in st.session_state:
    job_info = st.session_state.job_info
    st.write(f"**Job Title:** {job_info['job_title']}")
    st.write(f"**Total Candidates:** {job_info['total_candidates']}")
    
    # NEW: Option to upload additional candidate resumes
    st.markdown("""### You can add more candidate CVs here \n
                \n make sure the cvs are in pdf format, not images or images saved as pdf""")
    additional_files = st.file_uploader("Upload additional candidate CV PDFs (optional)", type=["pdf"], accept_multiple_files=True)

    ignore_auto_cvs = st.checkbox(
        "**Ignore Auto-Candidates (analyse only uploaded cvs)**", 
        value=False # Default state is unchecked (False)
    )


    additional = st.text_input(
        "Additional instruction", 
        placeholder="e.g., emphasize JavaScript and Django experience"
    )


    if st.button("🚀 Start Analysis"):
        try:
            # Using multipart form-data in requests; prepare data and files
            payload = {
                "job_id": str(job_id),
                "instruction": additional,
                "ignore_auto": str(ignore_auto_cvs)
            }
            files_payload = {}
            if additional_files:
                # Note: Use a list for the same key when sending multiple files.
                files_payload = [
                    ('candidate_files', (cv.name, cv.getvalue(), cv.type))
                    for cv in additional_files
                ]
            try:
                response = requests.post(
                    f"{BACKEND_URL}/analyze-cvs/",
                    data=payload,
                    files=files_payload if files_payload else None,
                    headers=headers,
                    allow_redirects=True
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Error during cv analysis: {e}")
            if response.status_code == 200:
                st.success("✅ Analysis process started!")
                st.info("To check the status or view results of the analysis, please visit the [Check Analysis](./check_analysis) page.")
            else:
                st.error(f"Failed to start analysis. Status: {response.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")