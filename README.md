# OCR PDF/Image to TXT Converter — README

A PyQt/PySide6-based desktop tool that uses a local **Qwen2-VL-OCR-Instruct** model to extract text from PDFs and images and save results as .txt (or Markdown) files.

---

## PDF to TXT Converter

**The model directory misses the model.safetensor file with model+weights+biases since it exceeds the 2 GB LFS limit. Please visit the repo on Hugginface and download it to ./models/Qwen2-VL-OCR-2B-Instruct if you plan to pack the binary.**

### Overview
This is a GUI-based application that converts PDF documents and images into plain text or Markdown using advanced Optical Character Recognition (OCR) techniques. The application leverages PySide6 (or PyQt6) for the user interface and a transformer model (Qwen2-VL-OCR-2B-Instruct) for image-to-text extraction.

### Features
- Load multiple PDF files for processing
- Select an output directory to save extracted text (TXT or Markdown)
- Process images from a selected directory (common formats)
- Choose processing device (CPU/GPU)
- Progress tracking with a progress bar and status messages
- Automatic detection/choice of device and optional light/dark theme support via ThemeColors
- Uses local `Qwen2-VL-OCR-2B-Instruct` model for text extraction (AutoProcessor + AutoModelForImageTextToText)
- Error handling and user-friendly notifications
- Outputs: each PDF page saved as page_N.txt (or .md), images saved as individual TXT files in an images/ folder
- PyInstaller-friendly resource_path() for bundling

---

## Dependencies
To run this application, install:

```bash
pip install PySide6 torch transformers Pillow pymupdf torch # select torch based on your HW and cuda version
```

- Install torch following official instructions for your CUDA version if you plan to use GPU.
- transformers must support trust_remote_code if you use the model's custom code API.

## Usage
1. Place the local model under models/Qwen2-VL-OCR-2B-Instruct or update the model path in load_model().
2. Run the application:

```
python ocr_gui.py
```

### GUI steps:
1. Click "Select PDF Files" to choose one or more PDFs (each PDF page becomes one TXT/MD file).
2. Click "Select Image Directory" to load supported images (.jpg/.jpeg/.png/.bmp/.tiff/.tif/.webp).
3. Click "Select Output Directory" to choose where TXT/MD files are written.
4. Choose processing device (cuda or cpu) from the dropdown.
5. Click "Start Conversion" to begin. The progress bar and status label update during processing.
6. Click "Reset" to clear selections.

### Output layout:
- For each PDF: a subfolder named after the PDF base name containing page_N.txt (or .md) files.
- For images: a folder images/ inside the output directory with one TXT file per image.

### Model Details:
- Model: Qwen2-VL-OCR-2B-Instruct (local directory expected)
- Loaded via:
```AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)```
```AutoModelForImageTextToText.from_pretrained(local_model_path, trust_remote_code=True)``
- Device/dtype:
- Uses torch.float16 and CUDA when available (to reduce memory)
- Falls back to torch.float32 on CPU
- The processor is used to create a chat-style prompt (apply_chat_template) combining image + instruction before tokenization and generation.

### Code Structure & Key Functions:
- resource_path(relative_path)
- Returns absolute path and supports PyInstaller by checking sys._MEIPASS.
- process_vision_info(messages)
- Extracts image and video inputs from chat-like message dicts and returns (image_inputs, video_inputs).
- StyleSheet / ThemeColors
- Centralized styles for QMainWindow, QFrame, QLabel, QPushButton, inputs, and QProgressBar using a colors dict. ThemeColors can provide light/dark palettes.
- MainWindow
- initUI(): Builds the UI (file selectors, device chooser, progress bar, buttons).
- select_pdf_files(): Opens QFileDialog to pick PDFs and lists them.
- select_image_dir(): Selects directory and gathers supported image files recursively.
- select_output_dir(): Chooses destination directory.
- load_model(): Loads processor and model from a local directory, sets device/dtype, moves model to device.
- process_inputs(): Validates selections, computes total pages, iterates PDFs and images, updates progress, and saves outputs.
- run_inference(img, output_dir, filename): Prepares messages, uses processor.apply_chat_template, tokenizes images and prompt, runs model.generate decodes generated token IDs, and writes result to file.

### Example adjustments:

Supported image extensions (change as needed):

```supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")```

Change local model directory:

```local_model_path = resource_path(os.path.join("models", "my-local-ocr-model"))```

## Packaging with PyInstaller:
- Install PyInstaller:

```pip install pyinstaller```

- Build executable (directory mode only, since model is larger > 4 GB); when not including model into executable: option --onefile is suitable:

```pyinstaller --onefile --noconfirm --name Qwen2_VL_2B_OCR --add-data $env:CONDA_PREFIX\Lib\site-packages\torch\lib;torch\lib" --add-data "$env:CONDA_PREFIX\Library\bin;cudalibs" --add-data "models;models" --hidden-import torch --hidden-import transformers --hidden-import PySide6 --hidden-import fitz --icon .\ocr.ico ocr_gui.py```

## License & Legal

- The application code provided in this repository is released under your chosen license. Include a LICENSE file (for example MIT, Apache-2.0) to state the permissions and conditions that apply to the application code itself.

- The Qwen2-VL-OCR-2B-Instruct model files used by this application are the property of their respective rights holder(s). **The model belongs to Qwen (the model provider) and is subject to Qwen’s license and terms of use.** You must comply with all license terms, usage restrictions, and attribution requirements specified by Qwen for the model.

- You are responsible for:
  - Obtaining the model files from an authorized source (for example, the official Qwen/HuggingFace repository or other official distribution channel).
  - Reviewing and following Qwen’s license, terms of service, and any export-control, privacy, or data-processing restrictions that apply to the model.
  - Ensuring that any redistribution of the model (bundled with your application or otherwise) is permitted by Qwen’s license; if redistribution is not permitted, do not include the model files in distributed binaries or packages.

- No warranty: This software is provided "as is", without warranty of any kind, express or implied. The authors and contributors are not liable for damages arising from use of the software.