import requests

def get_ai_cv_data(jobid, appid):
    url = "https://bdev.remoting.work/api/v1/jobdescription/ai-cv-data"
    params = {
        "jobid": jobid,
        "appid": appid
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()   # Will raise HTTPError if status is 4xx or 5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error making GET request:", e)
        return None

if __name__ == "__main__":
    data = get_ai_cv_data(jobid=164, appid=546)
    print("Response data:")
    print(data)
