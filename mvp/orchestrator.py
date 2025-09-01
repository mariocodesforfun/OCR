from typing import Dict, Any
from fastapi import UploadFile
from utils.pdf_processor import PDFProcessor
from utils.ocr_client import OCRClient
import tempfile



class OCROrchestrator:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ocr_client = OCRClient()

    # process ocr markdown
    def process_ocr_markdown(self, file: UploadFile) -> Dict[str, Any]:
        """
        Process a PDF or image file and return OCR results in Markdown format.
        """
        # Check file type
        content_type = file.content_type.lower()
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''

        if content_type == 'application/pdf' or file_extension == 'pdf':
            # Process as PDF
            processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
            image_paths = self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)
        else:
            # Process as image
            image_paths = self._save_uploaded_image(file)

        # Read the first image for OCR
        with open(image_paths[0], 'rb') as f:
            image_bytes = f.read()

        ocr_result = self.ocr_client.markdown_openai(image_bytes)

        return {
            "status": "success",
            "image_paths": image_paths,
            "ocr_result": ocr_result
        }

    def _save_uploaded_image(self, file: UploadFile) -> list:
        """Save uploaded image to temporary file and return path."""
        try:
            # Read file content
            file_content = file.file.read()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1] if file.filename else 'jpg'}") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            return [temp_path]

        except Exception as e:
            raise Exception(f"Error saving uploaded image: {str(e)}")
