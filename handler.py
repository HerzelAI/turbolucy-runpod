import os
import sys
import threading
from flask import Flask, jsonify, request

# Create Flask app immediately
app = Flask(__name__)

# Global state for status
model_loading_status = "idle"
loading_error = None
pipe = None
upscaler = None

# Health check endpoint (must be fast!)
@app.route('/ping', methods=['GET'])
@app.route('/health', methods=['GET'])
def ping():
    return jsonify({
        "status": "healthy" if model_loading_status == "ready" else "initializing",
        "model_status": model_loading_status,
        "error": loading_error
    })

# Main inference endpoint
@app.route('/', methods=['POST'])
@app.route('/run', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate():
    if model_loading_status != "ready":
        return jsonify({"error": "Model is still loading, please wait", "status": model_loading_status}), 503
    
    # We'll import the actual processing logic inside here to keep startup fast
    from processing import run_inference
    return run_inference(request, pipe, upscaler)

def background_setup():
    global model_loading_status, loading_error, pipe, upscaler
    try:
        model_loading_status = "loading"
        print("⏳ Background: Starting heavy imports and model loading...")
        
        # Heavy imports moved here
        import torch
        import numpy as np
        from PIL import Image
        from diffusers import AutoencoderKLWan, LucyEditPipeline
        from diffusers.utils import export_to_video, load_video
        
        # Patches
        _original_sdpa = torch.nn.functional.scaled_dot_product_attention
        def _patched_sdpa(*args, **kwargs):
            kwargs.pop("enable_gqa", None)
            return _original_sdpa(*args, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

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

        if MODEL_DIR is None:
            from huggingface_hub import snapshot_download
            target_dir = "/runpod-volume/Lucy-Edit-1.1-Dev" if os.path.exists("/runpod-volume") else "/checkpoints"
            os.makedirs(target_dir, exist_ok=True)
            MODEL_DIR = snapshot_download("decart-ai/Lucy-Edit-1.1-Dev", local_dir=target_dir, ignore_patterns=["*.md", "*.txt"])

        # Load models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=torch.float32)
        pipe = LucyEditPipeline.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=torch.bfloat16)
        pipe.to(device)
        
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
             pipe.text_encoder = pipe.text_encoder.float()

        # SageAttention
        try:
            import sageattention
            # Patch logic here if needed
            print("✅ SageAttention ready")
        except:
            print("⚠️ SageAttention skipped")

        # Upscaler
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.srvgg_arch import SRVGGNetCompact
            model_ups = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            upscaler = RealESRGANer(scale=4, model_path="/models/realesr-animevideov3.pth", model=model_ups, tile=400, tile_pad=10, half=True, device=device)
            print("✅ Upscaler ready")
        except:
            upscaler = None

        model_loading_status = "ready"
        print("✅ Models fully ready for inference!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loading_status = "error"
        loading_error = str(e)

if __name__ == "__main__":
    # Start background loader
    threading.Thread(target=background_setup, daemon=True).start()
    
    # Start Flask Server on PORT immediately
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 TurboLucy Fast-Starter listening on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
