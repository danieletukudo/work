import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")
headers = {
    "User-Agent": os.getenv("USER_AGENT"),
    "Referer": os.getenv("REFERER"),
    "Accept": os.getenv("ACCEPT"),
    "Accept-Language": os.getenv("ACCEPT_LANGUAGE"),
    "Connection": "keep-alive"
}

# --- Page Config and Title ---
st.set_page_config(page_title="Check Analysis", layout="centered")
st.markdown("<h1>Analysis Status and Results</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'status_data' not in st.session_state:
    st.session_state.status_data = None
if 'failed_candidates_data' not in st.session_state:
    st.session_state.failed_candidates_data = None
if 'show_failed' not in st.session_state:
    st.session_state.show_failed = False
if 'failed_page' not in st.session_state:
    st.session_state.failed_page = 0
if 'top_candidates_data' not in st.session_state:
    st.session_state.top_candidates_data = None

# --- User Inputs ---
job_id = st.text_input("Enter Job ID to check status/results", placeholder="Job ID")

if job_id:
    # Let users specify the number of top candidates to display
    top_n = st.number_input("Number of Top Candidates to Display", min_value=1, max_value=1000, value=10, step=1)
    
    col1, col2 = st.columns(2)
    
    # --- Check Status Button (Fetches data and stores in session state) ---
    if col1.button("🔄 Check Status"):
        with st.spinner("Fetching status..."):
            try:
                # Fetch status
                status_response = requests.get(f"{BACKEND_URL}/analysis-status/{job_id}", headers=headers, allow_redirects=True)
                status_response.raise_for_status()
                st.session_state.status_data = status_response.json()

                # Fetch failed candidates and save to session state
                failed_response = requests.get(f"{BACKEND_URL}/failed-candidates/{job_id}", headers=headers, allow_redirects=True)
                failed_response.raise_for_status()
                st.session_state.failed_candidates_data = failed_response.json()
                st.session_state.show_failed = False  # Reset the display toggle

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data: {e}")
                st.session_state.status_data = None
                st.session_state.failed_candidates_data = None
    
    # --- View Results Button (Fetches and displays top candidates) ---
    if col2.button("View Results"):
        with st.spinner("Fetching results..."):
            try:
                results_response = requests.get(f"{BACKEND_URL}/top-candidates/{job_id}?top={top_n}", headers=headers, allow_redirects=True)
                results_response.raise_for_status()
                st.session_state.top_candidates_data = results_response.json()

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching results: {e}")
                st.session_state.top_candidates_data = None

    # --- Display Logic (Uses data from session state) ---
    if st.session_state.status_data:
        status = st.session_state.status_data
        auto_proc = status.get('auto_processed_cvs', 0)
        manual_proc = status.get('manual_processed_cvs', 0)
        processed_total = auto_proc + manual_proc
        total_cvs = status.get('total_cvs', 1)
        progress = min(100, (processed_total / total_cvs) * 100)
        
        st.markdown("### Current Status")
        # st.progress(int(progress))
        st.info(
            f"Status: {status['status'].upper()}\n\n"
            # f"Processed: {processed_total}/{total_cvs} CVs "
            f"(Auto CVS: {auto_proc}, Manual CVs processed: {manual_proc})\n\n"
            # f"Progress: {progress:.1f}%"
        )
        if status['status'] == 'completed':
            st.success("✅ Analysis completed! You can now view the results.")
        elif status['status'] == 'failed':
            st.error("❌ Analysis failed!")

    if st.session_state.failed_candidates_data:
        failed_data = st.session_state.failed_candidates_data
        total_failed = failed_data.get('total_failed', 0)
        st.markdown(f"### ❌ Failed Candidates: {total_failed}")
        
        if total_failed > 0:
            show_failed = st.checkbox("Show Failed Candidates", value=st.session_state.show_failed)
            st.session_state.show_failed = show_failed
            
            if st.session_state.show_failed:
                failed_candidates = failed_data.get('failed_candidates', [])
                page = st.session_state.failed_page
                page_size = 10
                start_idx = page * page_size
                end_idx = start_idx + page_size
                display_candidates = failed_candidates[start_idx:end_idx]
                
                st.markdown(f"#### Displaying {start_idx + 1} to {min(end_idx, total_failed)} of {total_failed}")
                for candidate in display_candidates:
                    st.markdown(f"**Candidate ID:** {candidate.get('candidate_id')}")
                    st.markdown(f"**Failure Reason:** {candidate.get('reason')}")
                    st.markdown(f"**File Path:** {candidate.get('file_path', 'N/A')}")
                    st.markdown("---")
                    
                col_prev, col_next = st.columns(2)
                if page > 0:
                    if col_prev.button("Previous Page", key="prev_failed"):
                        st.session_state.failed_page -= 1
                        st.rerun()
                if end_idx < total_failed:
                    if col_next.button("Next Page", key="next_failed"):
                        st.session_state.failed_page += 1
                        st.rerun()

    if st.session_state.top_candidates_data:
        results = st.session_state.top_candidates_data
        candidates = results.get('candidates', [])
        if candidates:
            st.markdown("### 🏆 Top Candidates")
            for idx, candidate in enumerate(candidates, 1):
                with st.expander(f"#{idx} - Score: {candidate['fitness_score']:.2f}"):
                    st.write(f"**Name:** {candidate['name']}")
                    st.write(f"**Candidate ID:** {candidate['candidate_id']}")
                    st.write(f"**Email:** {candidate.get('user_email', candidate.get('email', 'N/A'))}")
                    st.write("**Analysis Summary:**")
                    st.write(candidate.get('analysis_summary', 'No summary available'))
        else:
            st.info("No results available for this job.")