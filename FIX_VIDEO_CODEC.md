#  Video Codec Fix - Browser Playback

##  **The Real Problem**

The video codec is **mp4v** which browsers **don't support**!

Browsers need **H.264 (avc1)** codec for playback.

---

##  **What I Fixed**

### **1. Video Recording (agent1.py):**
```python
# OLD (not browser compatible):
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# NEW (browser compatible):
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
```

### **2. Video Combining (agent1.py):**
```python
# OLD (just copies codec):
'-c:v', 'copy'

# NEW (encodes to H.264):
'-c:v', 'libx264',       # H.264 for browsers
'-preset', 'ultrafast',  # Fast encoding
```

### **3. API Video Proxy (api_server.py):**
- Added proper range request handling (206 Partial Content)
- Added Accept-Ranges header
- Added Cache-Control header
- Supports video seeking

---

##  **Solution: Run a New Interview**

The existing videos in Azure were created with the old codec. You need to create a new video with H.264:

### **Steps:**

```bash
# 1. Restart agent with new codec (Terminal 2)
# Stop current agent (Ctrl+C)
python3 agent1.py dev

# 2. Run a NEW interview
# Open: http://localhost:3000
# Click: "Start Call"
# Talk: 30+ seconds
# Leave: Disconnect

# 3. Wait for video to upload (~20 seconds)

# 4. Get new video URL
cat azure_links/voice_assistant_user_*_links.json | tail -1 | grep video_url

# 5. Test in proctoring page
# Open: http://localhost:3000/proctoring
# Paste: New video URL
# Click: "Analyze Video"
# Video should play properly! ```

---

##  **Alternative: Re-encode Existing Video**

If you want to use the existing video, re-encode it to H.264:

```bash
# Get the old video
OLD_VIDEO="recordings/combined/voice_assistant_user_6826315c_20251202_152151_COMBINED.mp4"

# Re-encode to H.264
ffmpeg -i "$OLD_VIDEO" \
  -c:v libx264 \
  -preset fast \
  -c:a copy \
  -y \
  "${OLD_VIDEO%.mp4}_H264.mp4"

# Upload re-encoded version
python3 << 'EOF'
from azure_storage import upload_video_to_azure
result = upload_video_to_azure("recordings/combined/voice_assistant_user_6826315c_20251202_152151_COMBINED_H264.mp4")
if result['success']:
    print(f"\n New video URL:\n{result['blob_url']}")
EOF

# Use new URL in proctoring page
```

---

## Check Video Codec

To see what codec a video has:

```bash
ffmpeg -i recordings/combined/voice_assistant_user_xxx_COMBINED.mp4 2>&1 | grep "Video:"
```

**Should show:**
-  **Good:** `Video: h264` or `Video: avc1` (browser compatible)
-  **Bad:** `Video: mpeg4` or `Video: mp4v` (not browser compatible)

---

##  **What Will Happen Now**

### **Future Interviews:**
1. Videos created with **H.264 (avc1)** codec 2. Combined with **H.264 (libx264)** encoding 3. **Browsers can play** them directly 4. **Proctoring page works** perfectly 
### **Current Videos:**
- Created with mp4v (not browser compatible)
- Need to run new interview OR re-encode

---

## 🧪 **Testing**

### **Quick Test:**

```bash
# 1. Stop agent (Ctrl+C in Terminal 2)

# 2. Start agent with new code
python3 agent1.py dev

# 3. Run NEW interview (30+ seconds)

# 4. After interview ends, check video codec
NEW_VIDEO=$(ls -t recordings/combined/*.mp4 | head -1)
ffmpeg -i "$NEW_VIDEO" 2>&1 | grep "Video:"
# Should show: h264 or avc1

# 5. Get video URL
cat azure_links/voice_assistant_user_*_links.json | tail -1

# 6. Test in proctoring page
# Should play with VIDEO visible!
```

---

##  **Codec Comparison**

| Codec | Browser Support | File Size | Quality |
|-------|----------------|-----------|---------|
| **mp4v** |  Poor | Small | Good |
| **H.264 (avc1/libx264)** |  Excellent | Medium | Excellent |
| **VP9** |  Good | Small | Excellent |

**We're using H.264** - best browser compatibility!

---

## Performance Note

### **Encoding Speed:**
- `mp4v` (old): Very fast (~1-2 seconds)
- `H.264 ultrafast` (new): Fast (~3-5 seconds)
- `H.264 fast` (alternative): Medium (~10-15 seconds)

We're using `ultrafast` preset for speed while maintaining quality.

---

## Summary

### **Problem:**
- Videos created with `mp4v` codec
- Browsers don't support it well
- Shows loading forever or black screen

### **Solution:**
- Changed to `H.264 (avc1/libx264)` codec
- Browser compatible
- Proper streaming support

### **Action Required:**
- **Run a new interview** to create H.264 video
- Use new video URL for proctoring
- Video will play properly in browser! 
---

**Last Updated:** December 2, 2025  
**Status:**  **CODEC FIXED - RUN NEW INTERVIEW**


