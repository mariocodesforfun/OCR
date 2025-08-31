from typing import Dict, Any, Optional
from fastapi import UploadFile
from utils.pdf_processor import PDFProcessor
from utils.ocr_client import OCRClient
import json


class OCROrchestrator:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ocr_client = OCRClient()

    def process_ocr_json(self, file: UploadFile, schema: Optional[dict] = None) -> Dict[str, Any]:
        """
        Process a PDF file and return OCR results in JSON format.

        Args:
            file: Uploaded PDF file - of type UploadFile which is a FastAPI file object

        Returns:
            Dictionary containing status, image paths, and OCR results
        """
        # Convert PDF to images
        processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
        image_paths = self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)

        # Read the first image for OCR processing
        with open(image_paths[0], 'rb') as f:
            image_bytes = f.read()

        # Perform OCR
        ocr_result = self.ocr_client.ocr_json(image_bytes, schema)

        # Print the OCR result for debugging
        print(f"OCR Result: {json.dumps(ocr_result, indent=2)}")
        print(f"Generated {len(image_paths)} images from PDF")

        return {
            "status": "success",
            "image_paths": image_paths,
            "ocr_result": ocr_result
        }
