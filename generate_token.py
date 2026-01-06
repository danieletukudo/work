import os
from livekit import api
from dotenv import load_dotenv
import uuid

load_dotenv(".env")

def generate_room_token():
    """Generate a room token for testing in LiveKit playground"""
    
    # Get credentials from environment
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([url, api_key, api_secret]):
        print(" Error: Missing LiveKit credentials in .env file")
        print("Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")
        return
    
    # Create a unique room name
    room_name = f"test-room-{uuid.uuid4().hex[:8]}"
    
    # Generate token
    token = api.AccessToken(api_key, api_secret) \
        .with_identity("playground-user") \
        .with_name("Playground Tester") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
    
    jwt_token = token.to_jwt()
    
    print("\n" + "="*70)
    print(" LiveKit Room Token Generated!")
    print("="*70)
    print(f"\n Room Name: {room_name}")
    print(f"\n🔗 WebSocket URL:")
    print(f"   {url}")
    print(f"\n🎫 Room Token:")
    print(f"   {jwt_token}")
    print("\n" + "="*70)
    print("\n Instructions:")
    print("1. Start your agent: python agent1.py dev")
    print("2. Go to: https://agents-playground.livekit.io")
    print("3. Paste the URL and Token above")
    print("4. Click 'Connect'")
    print("\n" + "="*70 + "\n")
    
    return {
        "url": url,
        "room": room_name,
        "token": jwt_token
    }

if __name__ == "__main__":
    generate_room_token()
