# Local Multimodal OCR Toolkit

A modular Python framework for running, switching, and benchmarking state-of-the-art Vision-Language Models (VLMs) for OCR tasks locally.

This project provides a unified interface to interact with various Transformer-based OCR engines (Florence-2, Qwen-VL, DeepSeek-OCR, etc.) without managing disparate API styles or conflicting dependencies. It operates entirely offline using PyTorch.

## Core Features

1.  **Unified API**: Interact with any model using a standard `reader.read(image)` call. The system handles preprocessing, prompting, and output parsing.
2.  **Multi-Architecture Support**: Seamless integration of Microsoft, Alibaba, Tencent, DeepSeek, and Paddle architectures in a single environment.
3.  **Benchmarking Suite**: A built-in scientific testing rig that compares load times, inference speeds, and accuracy using dual metrics (Structural Precision vs. Content Similarity).
4.  **Global CLI**: Access the toolkit from any terminal window or external application via a simple `ocr` command.
5.  **Pure PyTorch**: All engines are implemented using standard `transformers` libraries, avoiding C++ dependency conflicts (DLL hell).

## Supported Engines

The toolkit includes wrappers for the following models, configured in `config.yaml`:

| Engine | Model ID | Params | Role |
| :--- | :--- | :--- | :--- |
| **Florence-2** | `microsoft/Florence-2-large` | 0.7B | **Generalist**. Fast, highly accurate for standard text and captions. |
| **Qwen2-VL** | `Qwen/Qwen2-VL-2B-Instruct` | 2B | **Balanced**. Excellent instruction following and speed. |
| **Qwen3-VL** | `Qwen/Qwen3-VL-8B-Instruct` | 8B | **Reasoning**. SOTA performance for complex visual reasoning (High VRAM). |
| **DeepSeek-OCR** | `unsloth/DeepSeek-OCR` | 3B | **Dense Text**. Specialized in document density and markdown formatting. |
| **HunyuanOCR** | `lvyufeng/HunyuanOCR` | 1B | **End-to-End**. Strong multilingual support and full-page parsing. |
| **PaddleOCR-VL** | `lvyufeng/PaddleOCR-VL-0.9B` | 0.9B | **Compact**. Efficient VLM specialized for tables and charts. |
| **GOT-OCR 2.0** | `stepfun-ai/GOT-OCR-2.0-hf` | 0.6B | **Formatting**. Specialized in LaTeX math and sheet music. |
| **EasyOCR** | `easyocr` | N/A | **Baseline**. Traditional CRAFT+ResNet pipeline (Control variable). |

---

## Installation

### Prerequisites
*   Python 3.10 (Recommended)
*   NVIDIA GPU with CUDA 12.1+ support
*   ~20GB Disk Space (ONLY IF you are downloading ALL models)

### Setup
1.  **Clone or Create Directory**
    ```bash
    mkdir C:\Tools\OCR
    cd C:\Tools\OCR
    ```

2.  **Create Environment**
    ```bash
    conda create -n local_ocr python=3.10 -y
    conda activate local_ocr
    ```

3.  **Install PyTorch (Crucial Step)**
    *We must install the CUDA-enabled version explicitly before other dependencies.*
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install Toolkit Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Alternative: Docker Setup
If you prefer containerization, you can build the image locally.
```bash
docker build -t local-ocr .
# Run with local volume mount to persist downloaded models
docker run --gpus all -it --rm -v ${PWD}:/app local-ocr
```

---

## Use Case 1: Global CLI Tool (System-Wide Access)

You can configure this tool to run from *any* terminal window (PowerShell, CMD, Git Bash) without manually activating Conda environments.

1.  **Edit `ocr.bat`**:
    Open `ocr.bat` and ensure `PYTHON_EXE` points to your `local_ocr` python path.
    *(Run `python -c "import sys; print(sys.executable)"` inside your env to find it).*

2.  **Install**:
    Move `ocr.bat` to a folder in your System PATH (e.g., `C:\Windows\System32` or a custom `C:\Bin`).

3.  **Usage**:
    Now, from anywhere on your PC:
    ```powershell
    # Basic Read (Uses default engine)
    ocr "C:\Users\Desktop\invoice.png"

    # Save to file
    ocr image.jpg > output.txt

    # Specific Engine & Task
    ocr math.jpg --engine qwen2 --task formula
    ```

---

## Use Case 2: Integration (Call from other Apps)

To use this OCR toolkit inside another Python project (or Node/C# app) **without** installing heavy dependencies in that project, use the Subprocess wrapper pattern.

**Example `ocr_client.py` (Safe to drop in any project):**
```python
import subprocess
import json

OCR_BAT_PATH = "ocr" # Assumes ocr.bat is in PATH

def read_image(image_path, engine=None):
    cmd = [OCR_BAT_PATH, image_path, "--raw"]
    if engine:
        cmd.extend(["--engine", engine])
    
    # Run silently
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout) # Returns dict with 'text'
    else:
        raise Exception(f"OCR Failed: {result.stderr}")

# Usage
print(read_image("test.jpg")['text'])
```

---

## Use Case 3: Benchmarking System

The toolkit includes `benchmark.py` to scientifically compare models against a Ground Truth.

**Metrics:**
1.  **Structure % (Levenshtein)**: Measures exact layout/character precision.
2.  **Content % (Cosine)**: Measures "Bag of Words" semantic accuracy (ignoring layout).

**Run Benchmark:**
```bash
# Compare all enabled engines on a specific image
python benchmark.py inputs/test.jpg inputs/truth.txt
```

**Example Results:**
*Hardware: NVIDIA RTX 3070 | Input: Magic Card (Text + Layout)*

| Model | Load (s) | Infer (s) | Struct % | Content % |
| :--- | :--- | :--- | :--- | :--- |
| **EasyOCR (Baseline)** | **1.10** | **0.37** | 32.99 | 75.23 |
| **Florence-2** | 3.31 | 1.14 | **98.22** | 95.71 |
| **GOT-OCR 2.0** | 3.66 | 1.96 | 86.77 | 78.63 |
| **HunyuanOCR 1B** | 4.27 | 2.47 | 96.90 | 97.14 |
| **DeepSeek-OCR 3B** | 5.38 | 4.76 | 96.03 | 95.71 |
| **Qwen2-VL 2B** | 4.75 | 2.05 | 88.65 | 86.97 |
| **Qwen3-VL 8B** | 10.81 | 48.65 | 98.04 | **98.56** |

---

## Directory Structure

```text
C:/Tools/OCR/
├── config.yaml           # User configuration (Enable/Disable engines here)
├── ocr.bat               # System Launcher
├── cli.py                # CLI Entry Point
├── benchmark.py          # Benchmarking Tool
├── ocr.py                # Main Python Library
├── core/                 # System logic
├── engines/              # Model wrappers
└── models_cache/         # Hugging Face model weights 
```