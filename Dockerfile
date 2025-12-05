# Use Python 3.10 as the base (The "Golden Standard" for our stack)
FROM python:3.10-slim

# 1. Install system dependencies
# OpenCV requires libgl1 and glib. Git is useful for some HF internal clones.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first to leverage Docker caching
COPY requirements.txt .

# 4. Install PyTorch (CUDA 12.1) explicitly first
# We do this separate from requirements.txt to ensure the correct index URL is respected
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Set Environment Variables for Stability (Matches ocr.py hacks)
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    MKL_THREADING_LAYER=GNU \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PYTHONUNBUFFERED=1

# 7. Copy the rest of the application
# (We usually mount the volume at runtime, but this ensures code exists if not mounted)
COPY . .

# 8. Default Command (Run the benchmark on the test image)
CMD ["python", "benchmark.py", "test.jpg"]