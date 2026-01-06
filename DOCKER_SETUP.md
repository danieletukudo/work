# Docker Setup Guide

This project uses Docker and Docker Compose for containerized deployment.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

## Services

The docker-compose.yml includes three services:

1. **backend** - FastAPI server (port 5001)
2. **frontend** - Next.js application (port 3000)
3. **agent** - LiveKit agent (optional, runs separately)

## Building and Running

### Build all services:
```bash
docker-compose build
```

### Start all services:
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f
```

### Stop all services:
```bash
docker-compose down
```

### Rebuild and restart:
```bash
docker-compose up -d --build
```

## Individual Service Commands

### Backend only:
```bash
docker-compose up backend
```

### Frontend only:
```bash
docker-compose up frontend
```

### Agent only:
```bash
docker-compose up agent
```

## Volumes

The following directories are mounted as volumes to persist data:
- `recordings/` - Audio/video recordings
- `proctoring_reports/` - Proctoring analysis reports
- `proctoring_videos/` - Proctoring video files
- `video_cache/` - Cached video files
- `downloads/` - Downloaded files
- `azure_links/` - Azure storage links

## Accessing Services

- Frontend: http://localhost:3000
- Backend API: http://localhost:5001
- API Documentation: http://localhost:5001/docs

## Development

For development, you may want to mount the source code as volumes for hot-reloading:

```yaml
volumes:
  - .:/app  # For backend
  - ./agent-starter-react:/app  # For frontend
```

Note: This is not included in the default docker-compose.yml for production use.

## Troubleshooting

### Port conflicts
If ports 3000 or 5001 are already in use, modify the port mappings in docker-compose.yml:
```yaml
ports:
  - "3001:3000"  # Change host port
```

### Permission issues
If you encounter permission issues with volumes, ensure the directories exist and have proper permissions:
```bash
mkdir -p recordings proctoring_reports proctoring_videos video_cache downloads azure_links
chmod -R 755 recordings proctoring_reports proctoring_videos video_cache downloads azure_links
```

### Rebuild after dependency changes
If you update requirements.txt or package.json:
```bash
docker-compose build --no-cache
docker-compose up -d
```

