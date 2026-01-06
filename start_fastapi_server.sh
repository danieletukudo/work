#!/bin/bash

echo "================================================================================
 Starting FastAPI Server
================================================================================
"

# Check if FastAPI is installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null

if [ $? -ne 0 ]; then
    echo " FastAPI not installed"
    echo ""
    echo "Installing FastAPI dependencies..."
    pip install fastapi uvicorn[standard] pydantic
    echo ""
fi

echo " Dependencies installed"
echo ""

# Start server
echo "Starting FastAPI server on port 5001..."
echo ""
echo " Interactive API Docs: http://localhost:5001/docs"
echo " ReDoc Documentation: http://localhost:5001/redoc"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================================
"

python3 api_server.py

