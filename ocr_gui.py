import os 
import sys
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QFileDialog,
                               QLabel, QComboBox, QProgressBar,
                               QScrollArea, QFrame, QMessageBox, QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import fitz  # PyMuPDF

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resource_path(relative_path):
    """Get absolute path to resource, works fine for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def process_vision_info(messages):
    """Local version of process_vision_info."""
    image_inputs = []
    video_inputs = []
    for message in messages:
        if message["role"] != "user":
            continue
        for content in message["content"]:
            if content["type"] == "image":
                image_inputs.append(content["image"])
            elif content["type"] == "video":
                video_inputs.append(content["video"])
    return image_inputs, video_inputs


class ThemeColors:
    @staticmethod
    def is_dark_theme(app):
        background_color = app.palette().color(QPalette.Window)
        return background_color.lightness() < 128

    @staticmethod
    def get_theme(app):
        is_dark = ThemeColors.is_dark_theme(app)
        if is_dark:
            return {
                'background': '#1e1e1e',
                'surface': '#252526',
                'border': '#3e3e42',
                'text': '#ffffff',
                'text_secondary': '#cccccc',
                'primary': '#0098ff',
                'primary_hover': '#0088ee',
                'error': '#ff5555',
                'error_hover': '#ff4444',
                'success': '#50fa7b',
                'success_hover': '#40ea6b',
                'input_background': '#3c3c3c',
                'input_text': '#ffffff',
                'progress_background': '#3c3c3c',
                'progress_chunk': '#0098ff'
            }
        else:
            return {
                'background': '#f0f2f5',
                'surface': '#ffffff',
                'border': '#e1e4e8',
                'text': '#24292e',
                'text_secondary': '#586069',
                'primary': '#2ea44f',
                'primary_hover': '#2c974b',
                'error': '#d73a49',
                'error_hover': '#cb2431',
                'success': '#28a745',
                'success_hover': '#22863a',
                'input_background': '#ffffff',
                'input_text': '#24292e',
                'progress_background': '#f6f8fa',
                'progress_chunk': '#2ea44f'
            }


class StyleSheet:
    def __init__(self, colors):
        self.colors = colors

    @property
    def MAIN_WINDOW(self):
        return f"""
        QMainWindow {{
            background-color: {self.colors['background']};
        }}
        """

    @property
    def FRAME(self):
        return f"""
        QFrame {{
            background-color: {self.colors['surface']};
            border-radius: 8px;
            border: 1px solid {self.colors['border']};
        }}
        """

    @property
    def LABEL(self):
        return f"""
        QLabel {{
            color: {self.colors['text']};
        }}
        """

    @property
    def TITLE(self):
        return f"""
        QLabel {{
            color: {self.colors['text']};
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
        }}
        """

    @property
    def BUTTON(self):
        return f"""
        QPushButton {{
            background-color: {self.colors['primary']};
            color: {self.colors['text']};
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {self.colors['primary_hover']};
        }}
        """

    @property
    def INPUT(self):
        return f"""
        QSpinBox, QComboBox {{
            border: 1px solid {self.colors['border']};
            border-radius: 6px;
            padding: 5px;
            background: {self.colors['input_background']};
            color: {self.colors['input_text']};
        }}
        QSpinBox:focus, QComboBox:focus {{
            border-color: {self.colors['primary']};
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            background: {self.colors['input_background']};
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background: {self.colors['input_background']};
            color: {self.colors['input_text']};
            selection-background-color: {self.colors['primary']};
            selection-color: {self.colors['text']};
        }}
        """

    @property
    def PROGRESS(self):
        return f"""
        QProgressBar {{
            border: 1px solid {self.colors['border']};
            border-radius: 6px;
            text-align: center;
            background-color: {self.colors['progress_background']};
            color: {self.colors['text']};
        }}
        QProgressBar::chunk {{
            background-color: {self.colors['progress_chunk']};
            border-radius: 5px;
        }}
        """


class MainWindow(QMainWindow):
    def __init__(self, style_sheet):
        super().__init__()
        self.style_sheet = style_sheet
        self.input_files = []
        self.output_dir = ""
        self.image_dir = ""
        self.image_files = []
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.initUI()

    def load_model(self):
        try:
            logger.debug("Loading model and processor...")
            self.status_label.setText("Loading model... Please wait.")

            local_model_path = resource_path(os.path.join("models", "Qwen2-VL-OCR-2B-Instruct"))

            self.processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)

            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=dtype,
            device_map="auto"
            )

            logger.debug("Model and processor loaded successfully from local directory.")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            logger.error(f"Model loading failed: {e}")
            return False

    def initUI(self):
        self.setWindowTitle("OCR PDF/Image to TXT converter")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet(self.style_sheet.MAIN_WINDOW)

        # Main widget and scroll area
        main_widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(main_widget)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet(self.style_sheet.FRAME)
        header_layout = QVBoxLayout(header_frame)

        title = QLabel("PDF/Image to Text Converter")
        title.setStyleSheet(self.style_sheet.TITLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description = QLabel("Convert PDF documents and images to TXT using advanced OCR")
        description.setStyleSheet(self.style_sheet.LABEL)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        header_layout.addWidget(description)
        layout.addWidget(header_frame)

        # Input/Output Section
        io_frame = QFrame()
        io_frame.setStyleSheet(self.style_sheet.FRAME)
        io_layout = QGridLayout(io_frame)
        io_layout.setSpacing(15)

        # PDF files
        self.btn_input = QPushButton("Select PDF Files")
        self.btn_input.setStyleSheet(self.style_sheet.BUTTON)
        self.btn_input.clicked.connect(self.select_pdf_files)
        io_layout.addWidget(self.btn_input, 0, 0)
        self.file_count_label = QLabel("No PDF files selected")
        self.file_count_label.setStyleSheet(self.style_sheet.LABEL)
        io_layout.addWidget(self.file_count_label, 0, 1)

        # Image directory
        self.btn_imgdir = QPushButton("Select Image Directory")
        self.btn_imgdir.setStyleSheet(self.style_sheet.BUTTON)
        self.btn_imgdir.clicked.connect(self.select_image_dir)
        io_layout.addWidget(self.btn_imgdir, 1, 0)
        self.imgdir_label = QLabel("No directory selected")
        self.imgdir_label.setStyleSheet(self.style_sheet.LABEL)
        io_layout.addWidget(self.imgdir_label, 1, 1)

        self.file_list = QLabel()
        self.file_list.setWordWrap(True)
        self.file_list.setStyleSheet(self.style_sheet.LABEL)
        io_layout.addWidget(self.file_list, 2, 0, 1, 2)

        # Output
        self.btn_output = QPushButton("Select Output Directory")
        self.btn_output.setStyleSheet(self.style_sheet.BUTTON)
        self.btn_output.clicked.connect(self.select_output_dir)
        io_layout.addWidget(self.btn_output, 3, 0)
        self.output_label = QLabel("No directory selected")
        self.output_label.setStyleSheet(self.style_sheet.LABEL)
        io_layout.addWidget(self.output_label, 3, 1)

        layout.addWidget(io_frame)

        # Settings Section
        settings_frame = QFrame()
        settings_frame.setStyleSheet(self.style_sheet.FRAME)
        settings_layout = QGridLayout(settings_frame)
        settings_layout.setSpacing(15)
        device_label = QLabel("Processing Device:")
        device_label.setStyleSheet(self.style_sheet.LABEL)
        settings_layout.addWidget(device_label, 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
        self.device_combo.setStyleSheet(self.style_sheet.INPUT)
        settings_layout.addWidget(self.device_combo, 0, 1)
        layout.addWidget(settings_frame)

        # Progress Section
        progress_frame = QFrame()
        progress_frame.setStyleSheet(self.style_sheet.FRAME)
        progress_layout = QVBoxLayout(progress_frame)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(self.style_sheet.LABEL)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(self.style_sheet.PROGRESS)
        self.progress_bar.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_frame)

        # Control Buttons
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet(self.style_sheet.FRAME)
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setSpacing(15)
        self.start_button = QPushButton("Start Conversion")
        self.start_button.setStyleSheet(self.style_sheet.BUTTON)
        self.start_button.clicked.connect(self.process_inputs)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.style_sheet.colors['error']};
                color: {self.style_sheet.colors['text']};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.style_sheet.colors['error_hover']};
            }}
        """)
        self.reset_button.clicked.connect(self.reset_all)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.reset_button)
        layout.addWidget(buttons_frame)

    def select_pdf_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF Files",
            os.path.expanduser('~'),
            "PDF Files (*.pdf)"
        )
        if files:
            self.input_files = files
            self.file_count_label.setText(f"{len(files)} file(s) selected")
            self.file_list.setText("\n".join(files))

    def select_image_dir(self):
        dir_ = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            os.path.expanduser('~')
        )
        if dir_:
            self.image_dir = dir_
            self.imgdir_label.setText(dir_)
            supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".tif")
            self.image_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(dir_)
                for f in files
                if f.lower().endswith(supported_ext)
            ]
            self.file_list.setText(
                f"{len(self.image_files)} image(s) found in directory\n" +
                "\n".join(self.image_files[:10]) + ("\n..." if len(self.image_files) > 10 else "")
            )

    def select_output_dir(self):
        dir_ = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            os.path.expanduser('~')
        )
        if dir_:
            self.output_dir = dir_
            self.output_label.setText(dir_)

    def process_inputs(self):
        logger.debug("Starting PDF/Image processing")

        if not self.input_files and not self.image_files:
            QMessageBox.warning(self, "Warning", "Please select PDF files or an image directory first!")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory first!")
            return

        try:
            if self.model is None or self.processor is None:
                if not self.load_model():
                    return

            total_pages = sum(len(fitz.open(pdf)) for pdf in self.input_files)
            total_pages += len(self.image_files)
            self.progress_bar.setRange(0, total_pages)
            current_progress = 0

            # Process PDFs
            for pdf_path in self.input_files:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_subdir = os.path.join(self.output_dir, base_name)
                os.makedirs(output_subdir, exist_ok=True)

                doc = fitz.open(pdf_path)
                self.status_label.setText(f"Processing {base_name}...")

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    self.run_inference(img, output_subdir, f"page_{page_num+1}.txt")

                    current_progress += 1
                    self.progress_bar.setValue(current_progress)
                    QApplication.processEvents()

                doc.close()

            # Process Images
            if self.image_files:
                img_outdir = os.path.join(self.output_dir, "images")
                os.makedirs(img_outdir, exist_ok=True)
                self.status_label.setText("Processing images...")

                for idx, img_path in enumerate(self.image_files, 1):
                    img = Image.open(img_path).convert("RGB")
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    self.run_inference(img, img_outdir, f"{base_name}.txt")

                    current_progress += 1
                    self.progress_bar.setValue(current_progress)
                    QApplication.processEvents()

            self.status_label.setText("Conversion completed successfully!")
            QMessageBox.information(self, "Success", "Conversion completed successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            self.status_label.setText("Error occurred during conversion")

    def run_inference(self, img, output_dir, filename):
        logger.debug(f"Running inference on {filename}")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Please extract all text from this image."}
            ]
        }]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt],
            images=[img],
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.device_combo.currentText())

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].replace("<|im_end|>", "")

        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        del inputs, output_ids
        torch.cuda.empty_cache()

    def reset_all(self):
        self.input_files = []
        self.output_dir = ""
        self.image_dir = ""
        self.image_files = []
        self.file_count_label.setText("No PDF files selected")
        self.imgdir_label.setText("No directory selected")
        self.file_list.setText("")
        self.output_label.setText("No directory selected")
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    colors = ThemeColors.get_theme(app)
    style_sheet = StyleSheet(colors)
    window = MainWindow(style_sheet)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
