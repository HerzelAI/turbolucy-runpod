# Use RunPod base with CUDA 12.4 (supports newer GPUs)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 1. System Dependencies
RUN apt-get update && apt-get install -y git ffmpeg wget libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 2. Upgrade PyTorch to nightly for RTX 5090/Blackwell (sm_120) support
# PyTorch stable doesn't support sm_120 yet - need nightly builds
RUN pip install --upgrade pip setuptools && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# A. Frameworks and Core
RUN pip install --no-cache-dir --ignore-installed \
    diffusers>=0.30.0 \
    transformers \
    accelerate \
    huggingface_hub \
    runpod \
    flask \
    imageio \
    imageio-ffmpeg \
    scipy \
    pandas \
    einops \
    ftfy \
    requests \
    opencv-python-headless

# B. TurboDiffusion Acceleration (SageAttention)
RUN pip install --no-cache-dir sageattention

# C. Real-ESRGAN Upscaler
RUN pip install --no-cache-dir realesrgan basicsr>=1.4.2 facexlib gfpgan

# 3. Model Weight Preparation
# Setup directories for weights (supports volume or baking)
RUN mkdir -p /models /checkpoints

# Download Real-ESRGAN weights during build
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -O /models/realesr-animevideov3.pth

# OPTIONAL: Bake Lucy-Edit weights (Uncomment if you don't want to use a Network Volume)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('decart-ai/Lucy-Edit-1.1-Dev', local_dir='/checkpoints', ignore_patterns=['*.md', '*.txt'])"

# 4. Copy Handler
WORKDIR /app
COPY handler.py /app/handler.py

# 5. Start
CMD [ "python", "-u", "/app/handler.py" ]
