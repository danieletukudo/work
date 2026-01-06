# Docker Setup for LiveKit Services

This Docker setup runs both the FastAPI server (`api_server.py`) and the LiveKit agent (`agent1.py dev`) in a single container.

## Prerequisites

1. Docker and Docker Compose installed
2. A `.env` file with your LiveKit credentials:
   ```env
   LIVEKIT_URL=https://your-livekit-server-url
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   ```

   You may also need additional environment variables for Azure Storage and OpenAI if your agent uses them.

## Quick Start

### Using Docker Compose (Recommended)

1. Make sure you have a `.env` file in the project root with your LiveKit credentials.

2. Build and start the services:
   ```bash
   docker-compose up --build
   ```

3. The services will be available at:
   - API Server: http://localhost:5001
   - API Docs: http://localhost:5001/docs
   - Health Check: http://localhost:5001/health

4. To run in detached mode:
   ```bash
   docker-compose up -d
   ```

5. To view logs:
   ```bash
   docker-compose logs -f
   ```

6. To stop the services:
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. Build the image:
   ```bash
   docker build -t livekit-services .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name livekit-services \
     -p 5001:5001 \
     --env-file .env \
     -v $(pwd)/recordings:/app/recordings \
     -v $(pwd)/downloads:/app/downloads \
     -v $(pwd)/azure_links:/app/azure_links \
     -v $(pwd)/proctoring_reports:/app/proctoring_reports \
     -v $(pwd)/proctoring_videos:/app/proctoring_videos \
     -v $(pwd)/video_cache:/app/video_cache \
     livekit-services
   ```

3. View logs:
   ```bash
   docker logs -f livekit-services
   ```

4. Stop the container:
   ```bash
   docker stop livekit-services
   docker rm livekit-services
   ```

## Volumes

The following directories are mounted as volumes to persist data:
- `recordings/` - Video and audio recordings
- `downloads/` - Evaluation files
- `azure_links/` - Azure storage links
- `proctoring_reports/` - Proctoring analysis reports
- `proctoring_videos/` - Downloaded proctoring videos
- `video_cache/` - Cached video files

## Environment Variables

Required environment variables (in `.env` file):
- `LIVEKIT_URL` - Your LiveKit server URL
- `LIVEKIT_API_KEY` - Your LiveKit API key
- `LIVEKIT_API_SECRET` - Your LiveKit API secret

Optional (depending on your setup):
- Azure Storage credentials (if using Azure)
- OpenAI API key (if using OpenAI models)
- Other service credentials

## Troubleshooting

### Check if services are running
```bash
docker-compose ps
```

### View logs
```bash
docker-compose logs -f livekit-services
```

### Restart services
```bash
docker-compose restart
```

### Rebuild after code changes
```bash
docker-compose up --build
```

### Access container shell
```bash
docker-compose exec livekit-services /bin/bash
```

## Health Check

The API server includes a health check endpoint at `/health` that verifies LiveKit credentials are configured. The Docker Compose setup includes a healthcheck that monitors this endpoint.

