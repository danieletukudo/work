#!/bin/bash
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $API_PID $AGENT_PID 2>/dev/null || true
    wait $API_PID $AGENT_PID 2>/dev/null || true
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT

echo "========================================"
echo "Starting LiveKit services..."
echo "========================================"

# Start API server
echo "Starting API server on port 5001..."
python api_server.py &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait a moment for API server to start
sleep 2

# Start LiveKit agent
echo "Starting LiveKit agent..."
python agent1.py dev &
AGENT_PID=$!
echo "Agent started with PID: $AGENT_PID"

echo "========================================"
echo "Both services are running"
echo "API server: http://localhost:5001"
echo "API docs: http://localhost:5001/docs"
echo "========================================"

# Wait for both processes
wait $API_PID $AGENT_PID

