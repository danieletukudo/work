# Dockerfile for Python Backend (FastAPI + LiveKit Agent)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# LiveKit Agent Dependencies
RUN pip install livekit
RUN pip install livekit-agents
RUN pip install livekit-plugins-deepgram
RUN pip install livekit-plugins-elevenlabs
RUN pip install livekit-plugins-openai
RUN pip install livekit-plugins-silero

# Environment and utilities
RUN pip install python-dotenv
RUN pip install humanfriendly
RUN pip install aiohttp

# Video/Audio processing
RUN pip install opencv-python
RUN pip install numpy

# Azure Storage
RUN pip install azure-storage-blob

# FastAPI Server (modern async API framework)
RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install pydantic

# Flask (backup - api_server_flask_backup.py)
RUN pip install flask
RUN pip install flask-cors

# Protobuf (for LiveKit)
RUN pip install protobuf





# Copy application code
COPY *.py ./
COPY  ./
COPY prompt.py ./
COPY evaluation_formatter.py ./
COPY extract_qa.py ./
COPY azure_storage.py ./
COPY save_links.py ./
COPY gaze_direction_detector.py ./

# Create necessary directories
RUN mkdir -p recordings/audio recordings/video recordings/combined \
    proctoring_reports proctoring_videos video_cache downloads azure_links

# Expose FastAPI port
EXPOSE 5001

# Default command (can be overridden in docker-compose)
CMD ["python", "api_server.py"]

