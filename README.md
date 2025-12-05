# Local Multimodal OCR Toolkit

A modular Python framework for running, switching, and benchmarking state-of-the-art Vision-Language Models (VLMs) for OCR tasks locally.

This project provides a unified interface to interact with various Transformer-based OCR engines (Florence-2, Qwen-VL, DeepSeek-OCR, etc.) without managing disparate API styles or conflicting dependencies. After model download, it operates entirely offline using PyTorch.

## Core Features

1.  **Unified API**: Interact with any model using a standard `reader.read(image)` call. The system handles preprocessing, prompting, and output parsing.
2.  **Multi-Architecture Support**: Seamless integration of Microsoft, Alibaba, Tencent, DeepSeek architectures in a single environment.
3.  **Benchmarking Suite**: A built-in scientific testing rig that compares load times, inference speeds, and accuracy using dual metrics (Structural Precision vs. Content Similarity).
4.  **Pure PyTorch**: Most engines are implemented using standard `transformers` libraries, avoiding C++ dependency conflicts (DLL hell) common with mixed-framework installations.

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
*   Python 3.10 (Recommended for maximum compatibility)
*   NVIDIA GPU with CUDA 12.1+ support
*   ~20GB Disk Space (ONLY IF you are downloading ALL models)

### Setup
1.  **Create Environment**
    ```bash
    conda create -n local_ocr python=3.10 -y
    conda activate local_ocr
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Alternative: Docker Setup
If you prefer not to install Python/Conda locally, you can run the entire toolkit in a container.

1.  **Build the Image**
    ```bash
    docker build -t local-ocr .
    ```

2.  **Run with GPU Support**
    *Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).*
---

## Use Case 1: OCR Service

Use this as a library to integrate OCR into other local applications.

```python
import ocr

# 1. Initialize (Loads default engine from config.yaml)
reader = ocr.OCR()

# 2. Basic Read
result = reader.read("documents/invoice.jpg")
print(result['text'])

# 3. Switch Engines at Runtime
# Switch to Qwen for better reasoning/instruction following
reader.load_engine("qwen2") 
result = reader.read("documents/math_homework.jpg", task="formula")

# 4. Access Specific Features
# Some engines (DeepSeek, Qwen) support specific tasks
# Options: 'ocr' (default), 'markdown', 'table', 'formula', 'det' (coordinates)
reader.read("layout.png", task="markdown")
```

**Configuration:**
Edit `config.yaml` to change the default engine, adjust GPU quantization (float16/float32), or modify model paths.

---

## Use Case 2: Benchmarking System

The toolkit includes `benchmark.py` to scientifically compare models against a Ground Truth.

### Comparison Metrics
OCR accuracy is difficult to quantify with a single number. This suite uses two distinct NLP-minded metrics:
1.  **Structure % (Levenshtein Distance)**: Measures character-level precision. Penalizes missing newlines, spacing errors, and typos. High scores indicate exact layout reproduction.
2.  **Content % (Cosine Similarity)**: Measures semantic content (Bag of Words). Does not penalize layout changes. High scores indicate the correct words were found, even if the order was slightly shuffled.

### Running the Benchmark
You can run the benchmark on any image. Optionally, provide a text file containing the ground truth to generate accuracy scores.

```bash
# Option 1: Just run inference and see the output
python benchmark.py inputs/test.jpg

# Option 2: Run inference and score against a text file
python benchmark.py inputs/test.jpg inputs/ground_truth.txt
```

### Example Results
*Hardware: NVIDIA RTX 4090 | Input: Magic Card (Text + Layout)*

| Model | Load (s) | Infer (s) | Struct % | Content % |
| :--- | :--- | :--- | :--- | :--- |
| **EasyOCR (Baseline)** | **1.10** | **0.37** | 32.99 | 75.23 |
| **Florence-2** | 3.31 | 1.14 | **98.22** | 95.71 |
| **GOT-OCR 2.0** | 3.66 | 1.96 | 86.77 | 78.63 |
| **HunyuanOCR 1B** | 4.27 | 2.47 | 96.90 | 97.14 |
| **DeepSeek-OCR 3B** | 5.38 | 4.76 | 96.03 | 95.71 |
| **Qwen2-VL 2B** | 4.75 | 2.05 | 88.65 | 86.97 |
| **Qwen3-VL 8B** | 10.81 | 48.65 | 98.04 | **98.56** |

*Note: in our testing Qwen3 achieves the highest semantic accuracy but requires significantly more compute resources. Florence-2 offers the best balance of speed vs. structural accuracy.*

---

## Extensibility

To add a new model:
1.  **Create Engine**: Add a new file in `engines/` inheriting from `BaseOCREngine`. Implement `load()` and `process()`.
2.  **Update Config**: Add the model parameters to `config.yaml`.
3.  **Register**: Add the import switch to `core/factory.py`.

## Directory Structure

```text
C:/Tools/OCR/
├── config.yaml           # User configuration
├── ocr.py                # Main interface
├── benchmark.py          # CLI Benchmarking tool
├── requirements.txt      # Dependencies
├── core/                 # System logic (Factory, Scorer, Logger)
├── engines/              # Model wrappers (Florence, Qwen, etc.)
└── models_cache/         # Hugging Face model weights (Ignored by git)
```
