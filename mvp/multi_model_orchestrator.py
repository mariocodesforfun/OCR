from typing import Dict, Any
from fastapi import UploadFile
from utils.pdf_processor import PDFProcessor
from utils.ocr_client import OpenAiOCRProvider
from utils.json_extractor import JSONExtractor
import tempfile
import base64
import os
from providers.mistral_ocr_provider import MistralOCRProvider
from providers.gpt4o_provider import GPT4OProvider
from providers.gemini_segment import GeminiSegmentProvider

class MultiModelOCROrchestrator:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ocr_client = OpenAiOCRProvider()
        self.json_extractor = JSONExtractor()
        self.mistral_ocr_provider = MistralOCRProvider()
        self.gpt4o_provider = GPT4OProvider()
        self.gemini_segment_provider = GeminiSegmentProvider()

    def get_enhanced_json_ocr_result(self, file: UploadFile, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a PDF or image file using two-stage approach: Mistral + GPT-4o focused analysis.
        """
        content_type = file.content_type.lower()
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''

        if content_type == 'application/pdf' or file_extension == 'pdf':
            processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
            image_paths = self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)
        else:
            image_paths = self._save_uploaded_image(file)


        gpt_segmentation_prompt = self.ocr_client.gpt_segmentation_prompt(image_path=image_paths[0])
        print(gpt_segmentation_prompt)
        segementation_masks = self.gemini_segment_provider.extract_segmentation_masks(prompt=gpt_segmentation_prompt, image_path=image_paths[0])

        with open(image_paths[0], 'rb') as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        mistral_extracted_images = self.mistral_ocr_provider.process_mistral_ocr(image_base64)

        # Get the markdown from the results.txt file that Mistral provider writes
        results_path = os.path.join(os.path.dirname(__file__), "results.txt")
        mistral_markdown = ""
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                mistral_markdown = f.read()

        number_of_images = len(mistral_extracted_images)

        if mistral_extracted_images and number_of_images > 1:
            # Create a simple mapping for the OCR client (without context since we don't have it anymore)
            print("analyze with gpt-4o")
            image_paths_mapping = {path: {"context": ""} for path in mistral_extracted_images}
            analyzed_images = self.ocr_client.analyze_images_with_context(image_paths_mapping, schema)
            final_markdown = self._aggregate_image_results(mistral_markdown, analyzed_images)
        else:
            print("using only mistral ocr")
            final_markdown = mistral_markdown
            # final_markdown = self.gpt4o_provider.extract_markdown(image_bytes, schema)


        print("----------FINAL MARKDOWN----------------------")
        print(final_markdown)
        print("-----END OF FINAL MARKDOWN----------------------")

        enhanced_json_result = self.json_extractor.extract_json(final_markdown, schema)

        return enhanced_json_result

    def _save_uploaded_image(self, file: UploadFile) -> list:
        """Save uploaded image to temporary file and return path."""
        try:
            file_content = file.file.read()

            extension = 'jpg'
            if file.filename:
                clean_filename = file.filename.split('?')[0]
                if '.' in clean_filename:
                    ext = clean_filename.split('.')[-1].lower()
                    if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'pdf']:
                        extension = ext

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            return [temp_path]

        except Exception as e:
            raise Exception(f"Error saving uploaded image: {str(e)}")

    def _aggregate_image_results(self, original_markdown: str, analyzed_results: Dict[str, str]) -> str:
        """
        Aggregate only the GPT-4o analyzed content, removing original context and images.
        """
        final_markdown_parts = []

        for i, (image_path, analyzed_content) in enumerate(analyzed_results.items()):
            if analyzed_content and analyzed_content.strip():
                final_markdown_parts.append(analyzed_content.strip())

        return "\n\n".join(final_markdown_parts)