import runpod
import torch
import os
import sys
import time
import base64
import requests
import tempfile
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video

# --- GLOBAL PATCHES ---
# Fixes: TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'
_original_sdpa = torch.nn.functional.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

# --- ACTUAL TURBO: SAGE ATTENTION PROCESSOR ---
class WanSageAttentionProcessor:
    def __init__(self):
        try:
            import sageattention
            self.sageattn = sageattention.sageattn
            print("✅ WanSageAttentionProcessor: Using SageAttention kernel.")
        except ImportError:
            self.sageattn = None
            print("⚠️ WanSageAttentionProcessor: SageAttention not found, falling back to SDPA.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs):
        # 1. Project Q, K, V
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
        # 2. Reshape for attention (batch, head, seq, dim)
        query = query.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)

        # 3. Apply Query/Key Normalization (Wan specific)
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            query = attn.q_norm(query)
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            key = attn.k_norm(key)

        # 4. Apply Rotary Embeddings
        if rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # 5. Core Attention Calculation
        if self.sageattn is not None:
            # SageAttention is INT8 quantized internally
            # It expects (batch, head, seq, dim)
            hidden_states = self.sageattn(query, key, value)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        # 6. Reshape and Project Out
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * attn.head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states) # Dropout

        return hidden_states

def patch_wan_with_sage(transformer):
    print("🚀 Patching Transformer with explicit SageAttention processors...")
    processor = WanSageAttentionProcessor()
    transformer.set_attn_processor(processor)

# --- RUNTIME HOTFIX ---
# Fixes "NameError: name 'ftfy' is not defined" without rebuilding Docker
try:
    import ftfy
except ImportError:
    import subprocess
    print("📦 Installing missing dependency: ftfy...")
    subprocess.run([sys.executable, "-m", "pip", "install", "ftfy"], check=True)
    import ftfy
    print("✅ ftfy installed successfully")

# --- CONFIGURATION ---
# More robust path detection
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

# Global pipeline placeholders
pipe = None
upscaler = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants for video processing
FPS = 24
MAX_FRAMES_PER_PASS = 120  # 5 seconds at 24fps (Lucy-Edit works best with <=120 frames)

def load_models():
    global pipe, upscaler, MODEL_DIR
    
    if MODEL_DIR is None:
        print("⚠️ Weights not found in any standard path! Attempting to download...")
        from huggingface_hub import snapshot_download
        # Try to download to /runpod-volume if it exists, otherwise /checkpoints
        target_dir = "/runpod-volume/Lucy-Edit-1.1-Dev" if os.path.exists("/runpod-volume") else "/checkpoints"
        os.makedirs(target_dir, exist_ok=True)
        try:
            MODEL_DIR = snapshot_download(
                "decart-ai/Lucy-Edit-1.1-Dev", 
                local_dir=target_dir, 
                ignore_patterns=["*.md", "*.txt"]
            )
            print(f"✅ Downloaded weights to {MODEL_DIR}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to find or download weights: {e}")

    print(f"⏳ Loading Lucy-Edit-1.1-Dev from {MODEL_DIR}...")
    start_time = time.time()

    # 1. Optimizations: Enable SageAttention if available
    try:
        import sageattention
        print(f"🚀 SageAttention detected. Optimizing kernels...")
    except ImportError:
        print("ℹ️ SageAttention not found, using native Flash Attention.")

    # 2. Load VAE (Float32 required)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_DIR,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # 3. Load Pipeline (BFloat16)
    pipe = LucyEditPipeline.from_pretrained(
        MODEL_DIR,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)

    # 4. Global Torch Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 5. Ensure text encoder stays float32 to avoid CUDA embedding errors
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.float()
        print("✅ Text encoder set to float32 for compatibility")
    
    # 6. TURBO: Specialized Attention (graceful fallback if unavailable)
    try:
        patch_wan_with_sage(pipe.transformer)
    except Exception as e:
        print(f"⚠️ SageAttention patch failed (non-fatal): {e}")

    # 7. Load Real-ESRGAN Upscaler
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        model_ups = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        upscaler = RealESRGANer(
            scale=4, 
            model_path=UPSCALER_PATH, 
            model=model_ups, 
            tile=400, 
            tile_pad=10, 
            pre_pad=0, 
            half=True, 
            device=device
        )
        print("✅ Real-ESRGAN upscaler loaded.")
    except Exception as e:
        print(f"⚠️ Upscaler failed to load: {e}")
        upscaler = None

    # 8. Quick Warm-up (smaller, error-tolerant)
    print("🔥 Quick warmup...")
    try:
        dummy_video = [Image.new('RGB', (416, 240)) for _ in range(5)]  # Smaller
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pipe(prompt="warmup", video=dummy_video, num_inference_steps=1).frames[0]
    except Exception as e:
        print(f"⚠️ Warmup skipped (non-fatal): {e}")

    print(f"✅ Setup complete in {time.time() - start_time:.2f}s")

def download_video(url, dest_path):
    print(f"⬇️ Downloading video: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def process_segment(video_frames, prompt, negative_prompt, inf_w, inf_h, guidance_scale, num_inference_steps, seed):
    """Process a single video segment with the given parameters."""
    global pipe
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_frames = pipe(
            prompt=prompt,
            video=video_frames,
            negative_prompt=negative_prompt,
            height=inf_h,
            width=inf_w,
            num_frames=len(video_frames),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).frames[0]
    
    return output_frames

async def handler(job):
    """
    Async generator handler for RunPod queue management.
    Yields progress updates and returns aggregated result.
    """
    global pipe, upscaler
    job_input = job["input"]

    # --- INPUTS ---
    prompt = job_input.get("prompt")
    video_url = job_input.get("video_url")
    negative_prompt = job_input.get("negative_prompt", "")
    height = job_input.get("height", 480)
    width = job_input.get("width", 832)
    guidance_scale = job_input.get("guidance_scale", 5.0)
    num_inference_steps = job_input.get("num_inference_steps", 8)  # Turbo Default: 8 steps
    seed = job_input.get("seed", 42)
    upscale_to_1080p = job_input.get("upscale_to_1080p", False)
    # New: target_duration in seconds (auto, 5, or 10)
    target_duration = job_input.get("target_duration", "auto")

    if not video_url or not prompt:
        yield {"error": "Missing 'video_url' or 'prompt'"}
        return

    # --- PROGRESS: DOWNLOADING ---
    yield {"status": "downloading", "progress": 0.1}

    # --- PREPARE VIDEO ---
    temp_in = f"/tmp/{job['id']}_in.mp4"
    download_video(video_url, temp_in)
    
    video = load_video(temp_in)
    total_input_frames = len(video)
    
    # Calculate target frames based on duration
    if target_duration == "auto":
        # Use all frames up to reasonable limit
        target_frames = min(total_input_frames, MAX_FRAMES_PER_PASS)
    elif target_duration == 10:
        # 10 seconds = 240 frames at 24fps
        target_frames = min(total_input_frames, 240)
    else:
        # Default 5 seconds = 120 frames
        target_frames = min(total_input_frames, 120)
    
    video = video[:target_frames]
    
    # Aspect ratio check
    orig_w, orig_h = video[0].size
    inf_w, inf_h = width, height
    if orig_h > orig_w:  # Portrait
        inf_w, inf_h = min(width, height), max(width, height)
    
    video = [frame.resize((inf_w, inf_h)) for frame in video]
    
    yield {"status": "video_prepared", "progress": 0.2, "total_frames": len(video)}

    # --- INFERENCE: SINGLE OR TWO-PASS ---
    print(f"🎨 Editing with prompt: {prompt}")
    
    all_output_frames = []
    
    if len(video) <= MAX_FRAMES_PER_PASS:
        # Single pass - video fits in one segment
        yield {"status": "processing", "progress": 0.3, "segment": "1/1"}
        
        output_frames = process_segment(
            video, prompt, negative_prompt, inf_w, inf_h,
            guidance_scale, num_inference_steps, seed
        )
        all_output_frames = output_frames
        
        yield {"status": "processing_complete", "progress": 0.7}
    else:
        # Two-pass processing for 10-second videos
        # Split at midpoint with 5-frame overlap for smoother transitions
        mid = len(video) // 2
        overlap = 5
        
        segment1 = video[:mid + overlap]
        segment2 = video[mid - overlap:]
        
        # Process first segment
        yield {"status": "processing_segment_1", "progress": 0.3, "segment": "1/2", "frames": len(segment1)}
        
        output1 = process_segment(
            segment1, prompt, negative_prompt, inf_w, inf_h,
            guidance_scale, num_inference_steps, seed  # Same seed for consistency
        )
        
        yield {"status": "processing_segment_2", "progress": 0.5, "segment": "2/2", "frames": len(segment2)}
        
        # Process second segment with SAME seed for consistent style
        output2 = process_segment(
            segment2, prompt, negative_prompt, inf_w, inf_h,
            guidance_scale, num_inference_steps, seed  # Same seed for consistency
        )
        
        # Blend overlap region for seamless transition
        blend_start = len(output1) - overlap
        blended_overlap = []
        for i in range(overlap):
            alpha = i / overlap  # 0 to 1
            frame1 = np.array(output1[blend_start + i])
            frame2 = np.array(output2[i])
            blended = ((1 - alpha) * frame1 + alpha * frame2).astype(np.uint8)
            blended_overlap.append(Image.fromarray(blended))
        
        # Concatenate: first segment (without overlap) + blended + second segment (after overlap)
        all_output_frames = output1[:blend_start] + blended_overlap + output2[overlap:]
        
        yield {"status": "segments_merged", "progress": 0.7, "total_frames": len(all_output_frames)}

    # --- UPSCALING ---
    final_w, final_h = inf_w, inf_h
    if upscale_to_1080p:
        yield {"status": "upscaling", "progress": 0.75}
        print("✨ Upscaling to 1080p...")
        tgt_w, tgt_h = (1920, 1080) if inf_w > inf_h else (1080, 1920)
        final_w, final_h = tgt_w, tgt_h
        
        if upscaler:
            upscaled = []
            for i, frame in enumerate(all_output_frames):
                try:
                    res, _ = upscaler.enhance(np.array(frame), outscale=2)
                    res_pil = Image.fromarray(res).resize((tgt_w, tgt_h), Image.LANCZOS)
                    upscaled.append(res_pil)
                except:
                    upscaled.append(frame.resize((tgt_w, tgt_h), Image.LANCZOS))
                
                # Progress update every 20 frames
                if i % 20 == 0:
                    yield {"status": "upscaling", "progress": 0.75 + (0.15 * i / len(all_output_frames))}
            
            all_output_frames = upscaled
        else:
            all_output_frames = [f.resize((tgt_w, tgt_h), Image.LANCZOS) for f in all_output_frames]

    # --- ENCODING ---
    yield {"status": "encoding", "progress": 0.9}
    
    temp_out = f"/tmp/{job['id']}_out.mp4"
    export_to_video(all_output_frames, temp_out, fps=FPS)
    
    with open(temp_out, 'rb') as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')

    # Cleanup
    os.remove(temp_in)
    os.remove(temp_out)

    # --- FINAL RESULT ---
    yield {
        "status": "success",
        "progress": 1.0,
        "video_base64": video_base64,
        "resolution": f"{final_w}x{final_h}",
        "frames": len(all_output_frames),
        "duration_seconds": len(all_output_frames) / FPS
    }


# --- STARTUP: LOAD MODELS BEFORE ACCEPTING JOBS ---
print("🚀 TurboLucy Serverless Starting...")
print("⏳ Loading models at startup (not lazy loading)...")
load_models()
print("✅ Models ready - accepting jobs!")

# Start RunPod serverless with async generator support and aggregate streaming
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True  # Aggregates yielded results for /run and /runsync
})
