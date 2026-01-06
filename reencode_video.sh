#!/bin/bash

echo "================================================================================
 RE-ENCODING VIDEO TO H.264 FOR BROWSER PLAYBACK
================================================================================
"

# Input video
INPUT_VIDEO="recordings/combined/voice_assistant_user_6826315c_20251202_152151_COMBINED.mp4"
OUTPUT_VIDEO="recordings/combined/voice_assistant_user_6826315c_20251202_152151_COMBINED_H264.mp4"

# Check if input exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo " Input video not found: $INPUT_VIDEO"
    exit 1
fi

echo " Input: $INPUT_VIDEO"
echo " Output: $OUTPUT_VIDEO"
echo ""

# Show current codec
echo "Current codec:"
ffmpeg -i "$INPUT_VIDEO" 2>&1 | grep "Video:" | head -1
echo ""

# Re-encode to H.264
echo " Re-encoding to H.264 (browser-compatible)..."
echo ""

ffmpeg -i "$INPUT_VIDEO" \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -c:a copy \
  -y \
  "$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
    echo ""
    echo " Re-encoding complete!"
    echo ""
    
    # Show new codec
    echo "New codec:"
    ffmpeg -i "$OUTPUT_VIDEO" 2>&1 | grep "Video:" | head -1
    echo ""
    
    # Show file sizes
    OLD_SIZE=$(du -h "$INPUT_VIDEO" | cut -f1)
    NEW_SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
    echo "File sizes:"
    echo "  Old (mp4v): $OLD_SIZE"
    echo "  New (H.264): $NEW_SIZE"
    echo ""
    
    # Upload to Azure
    echo " Uploading to Azure..."
    python3 << 'PYTHON_EOF'
from azure_storage import upload_video_to_azure
result = upload_video_to_azure("recordings/combined/voice_assistant_user_6826315c_20251202_152151_COMBINED_H264.mp4")
if result['success']:
    print(f"\n Upload successful!")
    print(f"\n NEW VIDEO URL (use this in proctoring):")
    print(result['blob_url'])
    print("")
else:
    print(f"\n Upload failed: {result['error']}")
PYTHON_EOF
    
    echo "================================================================================
 DONE! Use the new URL in proctoring page
================================================================================"
    
else
    echo ""
    echo " Re-encoding failed!"
    exit 1
fi

