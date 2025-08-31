from typing import Dict, Any, Optional
from fastapi import UploadFile
from utils.pdf_processor import PDFProcessor
from utils.ocr_client import OCRClient
from utils.schema_converter import SchemaConverter
import json


class OCROrchestrator:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ocr_client = OCRClient()
        self.schema_converter = SchemaConverter()

    def process_ocr_json(self, file: UploadFile, schema: Optional[dict] = None) -> Dict[str, Any]:
        """
        Process a PDF or image file and return OCR results in JSON format.

        Args:
            file: Uploaded file - PDF or image (PNG, JPG, etc.)

        Returns:
            Dictionary containing status, image paths, and OCR results
        """
        # Check file type and handle accordingly
        filename = file.filename or ""
        content_type = file.content_type or ""
        
        if filename.lower().endswith('.pdf') or 'pdf' in content_type.lower():
            # Handle PDF files
            processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
            image_paths = self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)
            
            # Read the first image for OCR processing
            with open(image_paths[0], 'rb') as f:
                image_bytes = f.read()
        else:
            # Handle image files directly
            file.file.seek(0)  # Reset file pointer
            image_bytes = file.file.read()
            image_paths = ["direct_image"]  # Placeholder for image files

        # Stage 1: Perform comprehensive OCR extraction (no schema constraint)
        raw_ocr_result = self.ocr_client.ocr_json(image_bytes, schema=None)
        
        # Stage 2: Convert to target schema if provided
        if schema:
            print("🔄 Converting OCR output to target schema...")
            schema_compliant_result = self.schema_converter.convert_to_schema(raw_ocr_result, schema)
            final_result = schema_compliant_result

            print(f"Raw OCR Result: {json.dumps(raw_ocr_result, indent=2)}")
            print(f"Schema-Converted Result: {json.dumps(final_result, indent=2)}")
        else:
            final_result = raw_ocr_result
            print(f"OCR Result (no schema): {json.dumps(final_result, indent=2)}")
        
        if filename.lower().endswith('.pdf') or 'pdf' in content_type.lower():
            print(f"Generated {len(image_paths)} images from PDF")
        else:
            print(f"Processed image file directly")

        return {
            "status": "success",
            "image_paths": image_paths,
            "ocr_result": final_result,
            "raw_extraction": raw_ocr_result if schema else None  # Keep original for debugging
        }
