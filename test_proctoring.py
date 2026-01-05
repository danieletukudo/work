"""
Test script for video proctoring system
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:5001"

# Test video URL from Azure
TEST_VIDEO_URL = "https://remotingwork.blob.core.windows.net/uploads/videos/voice_assistant_user_43a509c1_20251202_023211_COMBINED.mp4?sv=2022-11-02&ss=b&srt=o&sp=rwdlactf&se=2029-06-07T20:38:13Z&st=2024-06-07T12:38:13Z&spr=https,http&sig=7zyNgPz1ZpFFlBVbnNjB%2Bj94f9ZrvJcdppaAVY9BUWs%3D"

def test_api_health():
    """Test if API server is running"""
    print("=" * 80)
    print("Testing API Health...")
    print("=" * 80)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print(" API Server is running")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f" API Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(" Could not connect to API server")
        print(f"   Make sure the API server is running on {API_BASE_URL}")
        print(f"   Start it with: python3 api_server.py")
        return False
    except Exception as e:
        print(f" Error: {e}")
        return False

def test_proctoring_analysis():
    """Test proctoring analysis endpoint"""
    print("\n" + "=" * 80)
    print("Testing Proctoring Analysis...")
    print("=" * 80)
    print(f"Video URL: {TEST_VIDEO_URL[:80]}...")
    print("")
    
    try:
        print(" Sending analysis request...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/api/proctoring/analyze",
            json={"video_url": TEST_VIDEO_URL},
            timeout=600  # 10 minutes timeout
        )
        
        elapsed = time.time() - start_time
        print(f"  Request completed in {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 80)
            print(" Analysis completed successfully!")
            print("=" * 80)
            print(f"Integrity Score: {result.get('integrity_score', 0):.2f}%")
            print(f"Video Duration: {result.get('duration', 0):.2f}s")
            print(f"Left Gaze: {result.get('left_gaze_duration', 0):.2f}s")
            print(f"Right Gaze: {result.get('right_gaze_duration', 0):.2f}s")
            print(f"Multiple Faces: {result.get('multiple_face_periods', 0)} periods")
            
            if result.get('warnings'):
                print("\n  Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            
            if result.get('report_text'):
                print("\n" + "=" * 80)
                print("Full Report:")
                print("=" * 80)
                print(result['report_text'])
            
            return True
        else:
            print(f"\n Analysis failed with status {response.status_code}")
            print(response.json())
            return False
            
    except requests.exceptions.Timeout:
        print("\n Request timed out (took longer than 10 minutes)")
        return False
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VIDEO PROCTORING SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: API Health
    if not test_api_health():
        print("\n Cannot proceed without running API server")
        return
    
    # Test 2: Proctoring Analysis
    print("\n Starting video analysis (this will take several minutes)...")
    test_proctoring_analysis()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()

