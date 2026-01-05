#!/bin/bash

echo "================================================================================
 TESTING INTERVIEW EVALUATION SYSTEM
================================================================================
"

echo "Testing API Server..."
response=$(curl -s http://localhost:5001/health)
if [[ $response == *"healthy"* ]]; then
    echo " API Server is running"
else
    echo " API Server is NOT running"
    echo "   Start it with: python3 api_server.py"
    exit 1
fi

echo ""
echo "Testing API Endpoints..."
curl -s http://localhost:5001/api/evaluations | python3 -m json.tool > /dev/null
if [ $? -eq 0 ]; then
    echo " /api/evaluations endpoint working"
else
    echo " /api/evaluations endpoint failed"
    exit 1
fi

echo ""
echo "Checking Evaluation Files..."
eval_count=$(ls -1 downloads/evaluation_*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$eval_count" -gt 0 ]; then
    echo " Found $eval_count evaluation file(s) in downloads/"
    echo ""
    echo "Latest evaluations:"
    ls -lt downloads/evaluation_*.json | head -3 | awk '{print "   ", $9}'
else
    echo "  No evaluation files yet (this is OK if you haven't run an interview)"
fi

echo ""
echo "Checking TXT Files..."
txt_count=$(ls -1 downloads/evaluation_*.txt 2>/dev/null | wc -l | tr -d ' ')
if [ "$txt_count" -gt 0 ]; then
    echo " Found $txt_count TXT file(s) - evaluation formatting working"
else
    echo "  No TXT files yet"
fi

echo ""
echo "================================================================================
SYSTEM STATUS:  ALL CHECKS PASSED
================================================================================
"

echo "
READY TO TEST:
--------------
1. Start Agent:
   python3 agent1.py dev

2. Start Frontend:
   cd agent-starter-react && npm run dev

3. Open Browser:
   http://localhost:3000

4. Run Interview and Verify:
 Process exits cleanly after disconnect
 Evaluation appears in downloads/ folder
 Frontend shows new evaluation
 Can download JSON and TXT files

5. Run Another Interview:
 New unique evaluation file created
 Process exits cleanly again
 Frontend shows BOTH evaluations

================================================================================
"

