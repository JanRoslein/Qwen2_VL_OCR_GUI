## PDF to TXT Converter

### Overview
This is a GUI-based application that converts PDF documents into Markdown text using advanced Optical Character Recognition (OCR) techniques. The application leverages PySide6 for the user interface and a transformer model from Hugging Face for text extraction from images.

### Features
- Load multiple PDF files for processing
- Select an output directory to save extracted text
- Choose processing device (CPU/GPU)
- Progress tracking with a progress bar
- Automatic detection of light/dark theme for UI
- Uses Qwen2-VL-OCR-2B-Instruct model for text extraction
- Error handling and user-friendly notifications

### Dependencies
fell To run this application, you need the following dependencies installed:

```bash
pip install PySide6 torch transformers Pillow pymupdf
```

### Usage

- you may create an executable file using PyInstaller:
```pip install pyinstaller
pyinstaller --onefile --windowed --add-data "your_model_directory;your_model_directory" main.py
```

- replace your_model_directory with the directory containing your model files or download them to default directory, you may need to change the model repo specification in the code.
- you shall find you executable in dist directory.

1. **Start the application**
   ```bash
   python ocr_gui.py
   ```

2. **Select PDF Files**
   - Click the "Select PDF Files" button to choose one or more PDFs.

3. **Choose Output Directory**
   - Click the "Select Output Directory" button and specify where the extracted text files should be saved.

4. **Select Processing Device**
   - Choose between "cuda" (GPU) and "cpu" for processing.

5. **Start Conversion**
   - Click the "Start Conversion" button to begin the process.
   - The progress bar will update as pages are processed.

6. **Reset the application**
   - Click the "Reset" button to clear selected files and output directories.

### Model Details
This application loads the `Qwen2-VL-OCR-2B-Instruct` model from Hugging Face for text extraction. The model is optimized for GPU inference but can also run on CPU.

### Code Structure
- `MainWindow`: The main GUI window handling file selection, progress tracking, and processing.
- `ThemeColors`: Handles dark/light mode themes.
- `StyleSheet`: Defines UI styles for buttons, labels, and progress bars.
- `process_pdfs()`: The core function that extracts text from PDF pages and saves them as Markdown.

### Troubleshooting
- **Error: "CUDA out of memory"**
  - Try switching to CPU mode.
  - If vRAM is plentiful, you may try increasing the pixmap size to get better results.
- **Application does not start**
  - Ensure dependencies are installed correctly.
  - Run with `python -m main.py` to check for import errors.

## License
This insignificant project is released under no licence do what ever you want.