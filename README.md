# Qwen2 VL OCR GUI - Updated Installation

## New Requirements
```bash
# Install system dependencies
sudo apt install poppler-utils  # For PDF rendering

# Python packages
pip install llama-cpp-python[server]>=0.2.72 \
            pdf2image>=1.16.3 \
            pillow>=10.3.0 \
            pymupdf>=1.24.1
```

## Model Setup
1. Create models directory:
```bash
mkdir -p models/
```
2. Download GGUF model:
```bash
wget https://huggingface.co/Qwen/Qwen2-VL-OCR-2B-Instruct-GGUF/resolve/main/Qwen2-VL-OCR-2B-Instruct.Q8_0.gguf \
     -O models/Qwen2-VL-OCR-2B-Instruct.Q8_0.gguf
```

## Architecture Changes
```python
# New model loading implementation
from llama_cpp import Llama

class MainWindow:
    def load_model(self):
        self.llm = Llama(
            model_path="./models/Qwen2-VL-OCR-2B-Instruct.Q8_0.gguf",
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")