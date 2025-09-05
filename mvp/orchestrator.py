from typing import Dict, Any
from fastapi import UploadFile
from utils.pdf_processor import PDFProcessor
from utils.ocr_client import OpenAiOCRProvider
from utils.json_extractor import JSONExtractor
import tempfile

class OCROrchestrator:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ocr_client = OpenAiOCRProvider()
        self.json_extractor = JSONExtractor()

    def process_ocr_json(self, file: UploadFile, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Process a PDF or image file and return OCR results as structured JSON."""
        image_paths = self._process_file(file)

        with open(image_paths[0], 'rb') as f:
            image_bytes = f.read()

        markdown_result = self.ocr_client.process_openai_ocr(image_bytes=image_bytes, schema=schema)
        json_result = self.json_extractor.extract_json(markdown_result, schema)

        return {
            "status": "success",
            "image_paths": image_paths,
            "json_result": json_result,
        }

    def process_ocr_markdown(self, file: UploadFile) -> Dict[str, Any]:
        """Process a PDF or image file and return OCR results as markdown."""
        image_paths = self._process_file(file)

        with open(image_paths[0], 'rb') as f:
            image_bytes = f.read()

        markdown_result = self.ocr_client.process_openai_ocr(image_bytes=image_bytes)
        return {
            "status": "success",
            "image_paths": image_paths,
            "markdown_result": markdown_result
        }

    def _process_file(self, file: UploadFile) -> list:
        """Process uploaded file (PDF or image) and return image paths."""
        content_type = file.content_type.lower()
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''

        if content_type == 'application/pdf' or file_extension == 'pdf':
            processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
            return self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)
        else:
            return self._save_uploaded_image(file)

    def _save_uploaded_image(self, file: UploadFile) -> list:
        """Save uploaded image to temporary file and return path."""
        try:
            file_content = file.file.read()

            # Extract clean file extension, handling URL parameters
            extension = 'jpg'  # default
            if file.filename:
                # Remove URL parameters (everything after ?)
                clean_filename = file.filename.split('?')[0]
                if '.' in clean_filename:
                    ext = clean_filename.split('.')[-1].lower()
                    # Validate extension
                    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'pdf']:
                        extension = ext

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            return [temp_path]

        except Exception as e:
            raise Exception(f"Error saving uploaded image: {str(e)}")
