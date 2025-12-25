import os
import base64
import time
import uuid
import requests
import numpy as np
from PIL import Image
from flask import jsonify

# Heavy imports are done inside the loading thread in handler.py
# Here we expect them to be available globally once models are ready
# or we import them locally inside the function.

FPS = 24
MAX_FRAMES_PER_PASS = 120

def run_inference(request, pipe, upscaler):
    # Local imports for heavy stuff
    import torch
    from diffusers.utils import export_to_video, load_video
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        data = request.get_json()
        job_input = data.get("input", data)
        
        prompt = job_input.get("prompt")
        video_url = job_input.get("video_url")
        negative_prompt = job_input.get("negative_prompt", "")
        height = job_input.get("height", 480)
        width = job_input.get("width", 832)
        guidance_scale = job_input.get("guidance_scale", 5.0)
        num_inference_steps = job_input.get("num_inference_steps", 8)
        seed = job_input.get("seed", 42)
        upscale_to_1080p = job_input.get("upscale_to_1080p", False)
        target_duration = job_input.get("target_duration", "auto")

        if not video_url or not prompt:
            return jsonify({"error": "Missing 'video_url' or 'prompt'"}), 400

        job_id = str(uuid.uuid4())[:8]
        temp_in = f"/tmp/{job_id}_in.mp4"
        
        # Download
        response = requests.get(video_url, stream=True, timeout=120)
        response.raise_for_status()
        with open(temp_in, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        video = load_video(temp_in)
        total_input_frames = len(video)

        # Duration logic
        if target_duration == 10:
            target_frames = min(total_input_frames, 240)
        elif target_duration == 5:
            target_frames = min(total_input_frames, 120)
        else:
            target_frames = min(total_input_frames, MAX_FRAMES_PER_PASS)
        video = video[:target_frames]

        # Resize for inference
        orig_w, orig_h = video[0].size
        inf_w, inf_h = width, height
        if orig_h > orig_w:
            inf_w, inf_h = min(width, height), max(width, height)
        video = [frame.resize((inf_w, inf_h)) for frame in video]

        def process_segment(frames, seed_val):
            generator = torch.Generator(device=device).manual_seed(seed_val)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = pipe(
                    prompt=prompt, video=frames, negative_prompt=negative_prompt,
                    height=inf_h, width=inf_w, num_frames=len(frames),
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator
                ).frames[0]
            return out

        # Inference
        if len(video) <= MAX_FRAMES_PER_PASS:
            all_output_frames = process_segment(video, seed)
        else:
            mid = len(video) // 2
            overlap = 5
            segment1 = video[:mid + overlap]
            segment2 = video[mid - overlap:]
            
            output1 = process_segment(segment1, seed)
            output2 = process_segment(segment2, seed)
            
            # Blend
            blend_start = len(output1) - overlap
            blended_overlap = []
            for i in range(overlap):
                alpha = i / overlap
                f1 = np.array(output1[blend_start + i])
                f2 = np.array(output2[i])
                blended = ((1 - alpha) * f1 + alpha * f2).astype(np.uint8)
                blended_overlap.append(Image.fromarray(blended))
            all_output_frames = output1[:blend_start] + blended_overlap + output2[overlap:]

        # Upscale
        final_w, final_h = inf_w, inf_h
        if upscale_to_1080p:
            tgt_w, tgt_h = (1920, 1080) if inf_w > inf_h else (1080, 1920)
            final_w, final_h = tgt_w, tgt_h
            if upscaler:
                upscaled = []
                for f in all_output_frames:
                    try:
                        res, _ = upscaler.enhance(np.array(f), outscale=2)
                        upscaled.append(Image.fromarray(res).resize((tgt_w, tgt_h), Image.LANCZOS))
                    except:
                        upscaled.append(f.resize((tgt_w, tgt_h), Image.LANCZOS))
                all_output_frames = upscaled
            else:
                all_output_frames = [f.resize((tgt_w, tgt_h), Image.LANCZOS) for f in all_output_frames]

        # Export
        temp_out = f"/tmp/{job_id}_out.mp4"
        export_to_video(all_output_frames, temp_out, fps=FPS)
        with open(temp_out, 'rb') as f:
            v_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        os.remove(temp_in)
        os.remove(temp_out)

        return jsonify({
            "status": "success",
            "video_base64": v_b64,
            "resolution": f"{final_w}x{final_h}",
            "frames": len(all_output_frames),
            "duration_seconds": len(all_output_frames) / FPS
        })
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return jsonify({"error": str(e)}), 500
