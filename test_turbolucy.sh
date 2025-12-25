#!/bin/bash

# REPLACE THESE VALUES
YOUR_API_KEY="PASTE_YOUR_RUNPOD_API_KEY_HERE"
ENDPOINT_ID="7rxxf1af25o8t8"

echo "🎯 Sending request to TurboLucy RunPod ($ENDPOINT_ID)..."

curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $YOUR_API_KEY" \
    -d '{
      "input": {
        "video_url": "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4",
        "prompt": "Change the apron to a yellow superhero suit with a black logo",
        "negative_prompt": "low quality, blurry, distorted",
        "num_frames": 49,
        "num_inference_steps": 15,
        "guidance_scale": 5.0,
        "upscale_to_1080p": true,
        "seed": 42
      }
    }'

echo -e "\n\n✅ Job submitted! Use the Job ID from the response to check status:"
echo "curl -X GET https://api.runpod.ai/v2/$ENDPOINT_ID/status/JOB_ID -H \"Authorization: Bearer \$YOUR_API_KEY\""
