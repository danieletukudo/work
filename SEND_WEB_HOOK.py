import requests
def send_evaluation_webhook(job_application_id: str, interview_score: float, video_url: str):
    """
    Send evaluation results to webhook endpoint.

    Args:
        job_application_id: The application ID (appid from API)
        interview_score: The interview score (0-100, typically overall_score * 10)
        video_url: The uploaded video URL
    """
    webhook_url = "https://bdev.remoting.work/api/v1/webhook/job-application/ai-interview-score"

    payload = {
        "job_application_id": int(job_application_id) if job_application_id and job_application_id.isdigit() else 0,
        "ai_interview_score": int(interview_score),
        "vid": video_url
    }

    try:

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return (f"Webhook sent successfully: {response.status_code}")

        else:
            return f"Webhook failed with status {response.status_code}: {response.text}"

    except Exception as e:
        return (f"Error sending webhook: {e}")

result = send_evaluation_webhook('123','90',"This is Daniel")

print(result)