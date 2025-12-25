# 🚀 TurboLucy on RunPod Serverless Guide

This repository adapts the accelerated Lucy-Edit (TurboLucy) for RunPod Serverless.

## Prerequisites
1. **Docker Hub**: You already have `arieliking` credentials.
2. **Network Volume**: Highly recommended to store `Lucy-Edit-1.1-Dev` weights (approx 10GB) to avoid heavy Docker images.

## Step 1: Transfer Files to RunPod Builder
Since you are on ARM, you must build this on an x86 machine (like a RunPod GPU pod).

**From your local machine:**
```powershell
cd c:\Users\altro\OneDrive\Desktop\runpod
scp -P 10359 -r turbolucy-runpod root@216.81.245.17:/workspace/
```

## Step 2: Build and Push
**On your RunPod pod:**
```bash
cd /workspace/turbolucy-runpod
chmod +x build_and_push.sh
./build_and_push.sh
```

## Step 3: Setup RunPod Serverless
1. **New Endpoint**: Create a new Serverless endpoint in RunPod.
2. **Image**: `arieliking/turbolucy-runpod:v1`
3. **GPU**: Select **A100** or **H100** (24GB+ VRAM).
4. **Volume**: Attach your **Network Volume** at `/runpod-volume`.
   - *Ensure the model weights are inside `/runpod-volume/Lucy-Edit-1.1-Dev`.*
   - *If not using a volume, you must uncomment the weight baking line in the Dockerfile and rebuild.*

## Step 4: Test Payload
Send this JSON to your endpoint:
```json
{
  "input": {
    "video_url": "https://example.com/video.mp4",
    "prompt": "Change the background to a sunny beach",
    "num_frames": 81,
    "num_inference_steps": 15,
    "upscale_to_1080p": true
  }
}
```
