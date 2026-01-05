

def read_job_description_example():
    """
    Reads a job description example from a markdown file.
    """
    file_path = "job_posting_agent/job_description_example.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    return content