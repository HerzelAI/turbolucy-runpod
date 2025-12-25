import torch
import os
import sys
import time
import base64
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video
import threading

# --- GLOBAL PATCHES ---
_original_sdpa = torch.nn.functional.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

# --- SAGE ATTENTION PROCESSOR ---
class WanSageAttentionProcessor:
    def __init__(self):
        try:
            import sageattention
            self.sageattn = sageattention.sageattn
            print("✅ SageAttention enabled")
        except ImportError:
            self.sageattn = None
            print("⚠️ SageAttention not found, using SDPA")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        query = query.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            query = attn.q_norm(query)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            key = attn.k_norm(key)
        if rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        if self.sageattn is not None:
            hidden_states = self.sageattn(query, key, value)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * attn.head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def patch_wan_with_sage(transformer):
    print("🚀 Patching with SageAttention...")
    processor = WanSageAttentionProcessor()
    transformer.set_attn_processor(processor)

# --- RUNTIME HOTFIX ---
try:
    import ftfy
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "ftfy"], check=True)
    import ftfy

# --- CONFIGURATION ---
POSSIBLE_PATHS = [
    os.environ.get("MODEL_DIR", ""),
    "/runpod-volume/Lucy-Edit-1.1-Dev",
    "/checkpoints",
    "/workspace/Lucy-Edit-1.1-Dev"
]
MODEL_DIR = None
for p in POSSIBLE_PATHS:
    if p and os.path.exists(os.path.join(p, "vae/config.json")):
        MODEL_DIR = p
        print(f"✅ Found weights at: {MODEL_DIR}")
        break

UPSCALER_PATH = "/models/realesr-animevideov3.pth"
pipe = None
upscaler = None
device = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 24
MAX_FRAMES_PER_PASS = 120

# Flask app
app = Flask(__name__)

def load_models():
    global pipe, upscaler, MODEL_DIR
    if MODEL_DIR is None:
        from huggingface_hub import snapshot_download
        target_dir = "/runpod-volume/Lucy-Edit-1.1-Dev" if os.path.exists("/runpod-volume") else "/checkpoints"
        os.makedirs(target_dir, exist_ok=True)
        MODEL_DIR = snapshot_download("decart-ai/Lucy-Edit-1.1-Dev", local_dir=target_dir, ignore_patterns=["*.md", "*.txt"])
        print(f"✅ Downloaded weights to {MODEL_DIR}")

    print(f"⏳ Loading Lucy-Edit from {MODEL_DIR}...")
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=torch.float32)
    pipe = LucyEditPipeline.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.float()
    try:
        patch_wan_with_sage(pipe.transformer)
    except Exception as e:
        print(f"⚠️ SageAttention patch failed: {e}")
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        model_ups = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        upscaler = RealESRGANer(scale=4, model_path=UPSCALER_PATH, model=model_ups, tile=400, tile_pad=10, pre_pad=0, half=True, device=device)
        print("✅ Upscaler loaded")
    except Exception as e:
        print(f"⚠️ Upscaler failed: {e}")
        upscaler = None

def download_video(url, dest_path):
    print(f"⬇️ Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def process_segment(video_frames, prompt, negative_prompt, inf_w, inf_h, guidance_scale, num_inference_steps, seed):
    global pipe
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_frames = pipe(
            prompt=prompt, video=video_frames, negative_prompt=negative_prompt,
            height=inf_h, width=inf_w, num_frames=len(video_frames),
            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator
        ).frames[0]
    return output_frames

# --- STARTUP LOGIC ---
model_loading_status = "idle"
loading_error = None

def model_loader_thread():
    global pipe, upscaler, model_loading_status, loading_error
    try:
        model_loading_status = "loading"
        load_models()
        model_loading_status = "ready"
        print("✅ Models ready - Background loading complete!")
    except Exception as e:
        print(f"❌ Critical Error loading models: {e}")
        model_loading_status = "error"
        loading_error = str(e)

# --- HEALTH CHECK ENDPOINTS ---
@app.route('/ping', methods=['GET'])
@app.route('/health', methods=['GET'])
def ping():
    return jsonify({
        "status": "healthy" if model_loading_status == "ready" else "initializing",
        "model_status": model_loading_status,
        "error": loading_error
    })

# --- MAIN INFERENCE ENDPOINT ---
@app.route('/', methods=['POST'])
@app.route('/run', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate():
    global pipe, upscaler
    
    if model_loading_status != "ready":
        return jsonify({"error": "Model is still loading, please wait", "status": model_loading_status}), 503

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

        import uuid
        job_id = str(uuid.uuid4())[:8]
        temp_in = f"/tmp/{job_id}_in.mp4"
        download_video(video_url, temp_in)
        video = load_video(temp_in)
        total_input_frames = len(video)

        if target_duration == 10:
            target_frames = min(total_input_frames, 240)
        elif target_duration == 5:
            target_frames = min(total_input_frames, 120)
        else:
            target_frames = min(total_input_frames, MAX_FRAMES_PER_PASS)
        video = video[:target_frames]

        orig_w, orig_h = video[0].size
        inf_w, inf_h = width, height
        if orig_h > orig_w:
            inf_w, inf_h = min(width, height), max(width, height)
        video = [frame.resize((inf_w, inf_h)) for frame in video]

        if len(video) <= MAX_FRAMES_PER_PASS:
            all_output_frames = process_segment(video, prompt, negative_prompt, inf_w, inf_h, guidance_scale, num_inference_steps, seed)
        else:
            mid = len(video) // 2
            overlap = 5
            segment1 = video[:mid + overlap]
            segment2 = video[mid - overlap:]
            output1 = process_segment(segment1, prompt, negative_prompt, inf_w, inf_h, guidance_scale, num_inference_steps, seed)
            output2 = process_segment(segment2, prompt, negative_prompt, inf_w, inf_h, guidance_scale, num_inference_steps, seed)
            blend_start = len(output1) - overlap
            blended_overlap = []
            for i in range(overlap):
                alpha = i / overlap
                frame1 = np.array(output1[blend_start + i])
                frame2 = np.array(output2[i])
                blended = ((1 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
                blended_overlap.append(Image.fromarray(blended))
            all_output_frames = output1[:blend_start] + blended_overlap + output2[overlap:]

        final_w, final_h = inf_w, inf_h
        if upscale_to_1080p:
            tgt_w, tgt_h = (1920, 1080) if inf_w > inf_h else (1080, 1920)
            final_w, final_h = tgt_w, tgt_h
            if upscaler:
                upscaled = []
                for frame in all_output_frames:
                    try:
                        res, _ = upscaler.enhance(np.array(frame), outscale=2)
                        upscaled.append(Image.fromarray(res).resize((tgt_w, tgt_h), Image.LANCZOS))
                    except:
                        upscaled.append(frame.resize((tgt_w, tgt_h), Image.LANCZOS))
                all_output_frames = upscaled
            else:
                all_output_frames = [f.resize((tgt_w, tgt_h), Image.LANCZOS) for f in all_output_frames]

        temp_out = f"/tmp/{job_id}_out.mp4"
        export_to_video(all_output_frames, temp_out, fps=FPS)
        with open(temp_out, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove(temp_in)
        os.remove(temp_out)

        return jsonify({
            "status": "success",
            "video_base64": video_base64,
            "resolution": f"{final_w}x{final_h}",
            "frames": len(all_output_frames),
            "duration_seconds": len(all_output_frames) / FPS
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- STARTUP ---
if __name__ == "__main__":
    print("🚀 TurboLucy HTTP Server Starting...")
    threading.Thread(target=model_loader_thread, daemon=True).start()
    
    port = int(os.environ.get("PORT", 8000))
    health_port = int(os.environ.get("PORT_HEALTH", port))
    
    print(f"✅ Web Server starting on port {port}")
    
    if health_port != port:
        print(f"🏥 Starting dedicated Health server on port {health_port}")
        from flask import Flask as HealthFlask
        health_app = HealthFlask("health")
        @health_app.route('/ping')
        @health_app.route('/health')
        def h_ping():
            return jsonify({
                "status": "healthy" if model_loading_status == "ready" else "initializing",
                "model_status": model_loading_status
            })
        threading.Thread(target=lambda: health_app.run(host="0.0.0.0", port=health_port), daemon=True).start()

    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
