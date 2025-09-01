import tempfile
from typing import Dict, Any, List, Optional
from fastapi import UploadFile

from providers.base_ocr_provider import BaseOCRProvider
from .markdown_comparator import MarkdownComparator, Disagreement
from utils.json_extractor import JSONExtractor
from utils.pdf_processor import PDFProcessor


class EnsembleOrchestrator:
    """Orchestrates the 4-step ensemble pipeline for OCR processing"""

    def __init__(self,
                 primary_provider: BaseOCRProvider,
                 secondary_provider: BaseOCRProvider,
                 adjudicator_provider: Optional[BaseOCRProvider] = None):
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.adjudicator_provider = adjudicator_provider or primary_provider
        self.comparator = MarkdownComparator()
        self.json_extractor = JSONExtractor()
        self.pdf_processor = PDFProcessor()

    @classmethod
    def create_default(cls):
        """Create ensemble orchestrator with default GPT-4O + Gemini providers"""
        try:
            from providers.gpt4o_provider import GPT4OProvider
            from providers.gemini_provider import GeminiProvider

            gpt4o_provider = GPT4OProvider()
            gemini_provider = GeminiProvider()

            return cls(
                primary_provider=gpt4o_provider,
                secondary_provider=gemini_provider,
                adjudicator_provider=gpt4o_provider
            )
        except Exception as e:
            raise Exception(f"Failed to initialize ensemble providers: {str(e)}")

    def process_ensemble_extraction(self, file: UploadFile, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the 4-step ensemble pipeline"""

        try:
            # Step 0: Prepare image
            image_bytes = self._prepare_image(file)

            # Step 1: Dual-Model OCR
            primary_markdown = self.primary_provider.extract_markdown(image_bytes)
            secondary_markdown = self.secondary_provider.extract_markdown(image_bytes)

            # Step 2: Binary Disagreement Detection
            disagreements = self.comparator.detect_disagreements(primary_markdown, secondary_markdown)
            has_disagreements = self.comparator.has_significant_disagreements(disagreements)

            # Step 3: Simple Resolution
            if has_disagreements:
                final_markdown = self._adjudicate_disagreements(
                    image_bytes, primary_markdown, secondary_markdown, disagreements
                )
                resolution_method = "adjudication"
            else:
                final_markdown = self._select_best_markdown(primary_markdown, secondary_markdown)
                resolution_method = "selection"

            # Step 4: Single JSON Extraction
            extracted_json = self.json_extractor.extract_json(final_markdown, schema)

            return {
                "status": "success",
                "ensemble_results": {
                    "primary_markdown": primary_markdown,
                    "secondary_markdown": secondary_markdown,
                    "disagreements": [self._disagreement_to_dict(d) for d in disagreements],
                    "disagreement_count": len(disagreements),
                    "has_significant_disagreements": has_disagreements,
                    "resolution_method": resolution_method,
                    "final_markdown": final_markdown,
                    "extracted_json": extracted_json
                },
                "providers_used": {
                    "primary": self.primary_provider.provider_name,
                    "secondary": self.secondary_provider.provider_name,
                    "adjudicator": self.adjudicator_provider.provider_name if has_disagreements else None
                },
                "processing_stats": {
                    "primary_length": len(primary_markdown),
                    "secondary_length": len(secondary_markdown),
                    "final_length": len(final_markdown)
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Ensemble extraction failed: {str(e)}",
                "providers_used": {
                    "primary": self.primary_provider.provider_name,
                    "secondary": self.secondary_provider.provider_name
                }
            }

    def _prepare_image(self, file: UploadFile) -> bytes:
        """Prepare image bytes from uploaded file"""
        content_type = file.content_type.lower() if file.content_type else ""
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''

        if content_type == 'application/pdf' or file_extension == 'pdf':
            # Process PDF
            processed_pdf_bytes = self.pdf_processor.preprocess_pdf(file)
            image_paths = self.pdf_processor._pdf_bytes_to_images(processed_pdf_bytes)

            # Use first page for now
            with open(image_paths[0], 'rb') as f:
                return f.read()
        else:
            # Process image directly
            image_path = self._save_uploaded_image(file)
            with open(image_path, 'rb') as f:
                return f.read()

    def _save_uploaded_image(self, file: UploadFile) -> str:
        """Save uploaded image to temporary file and return path"""
        file_content = file.file.read()

        # Extract safe file extension, handling long URLs
        if file.filename:
            # Handle URLs with query parameters - extract just the extension before '?'
            filename_clean = file.filename.split('?')[0]
            if '.' in filename_clean:
                extension = filename_clean.split('.')[-1]
            else:
                extension = 'jpg'
        else:
            extension = 'jpg'

        # Limit extension length to prevent filesystem issues
        extension = extension[:10] if extension else 'jpg'

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}"
        ) as temp_file:
            temp_file.write(file_content)
            return temp_file.name

    def _adjudicate_disagreements(self,
                                image_bytes: bytes,
                                primary_markdown: str,
                                secondary_markdown: str,
                                disagreements: List[Disagreement]) -> str:
        """Use vision-based adjudication for disagreements"""

        adjudication_prompt = self._build_adjudication_prompt(
            primary_markdown, secondary_markdown, disagreements
        )

        try:
            return self.adjudicator_provider.extract_markdown_with_context(
                image_bytes, adjudication_prompt
            )
        except Exception as e:
            # Fallback to selection if adjudication fails
            return self._select_best_markdown(primary_markdown, secondary_markdown)

    def _build_adjudication_prompt(self,
                                 primary_markdown: str,
                                 secondary_markdown: str,
                                 disagreements: List[Disagreement]) -> str:
        """Build a detailed prompt for adjudication"""

        prompt = """You are tasked with resolving disagreements between two OCR models.
Please extract the most accurate markdown representation of the image.

DISAGREEMENTS DETECTED:
"""

        for i, disagreement in enumerate(disagreements, 1):
            prompt += f"\n{i}. {disagreement.type.value.upper()} at {disagreement.location}:"
            prompt += f"\n   Model 1: {disagreement.provider1_content}"
            prompt += f"\n   Model 2: {disagreement.provider2_content}"
            prompt += f"\n   Confidence: {disagreement.confidence:.2f}\n"

        prompt += f"""
PRIMARY MODEL OUTPUT:
{primary_markdown[:500]}{'...' if len(primary_markdown) > 500 else ''}

SECONDARY MODEL OUTPUT:
{secondary_markdown[:500]}{'...' if len(secondary_markdown) > 500 else ''}

Please provide the most accurate markdown extraction, paying special attention to the disagreements listed above.
Focus on preserving exact numbers, table structures, and text content as they appear in the image.
"""

        return prompt

    def _select_best_markdown(self, primary_markdown: str, secondary_markdown: str) -> str:
        """Select the better markdown when no significant disagreements exist"""
        # Simple heuristic: choose the longer output (more detailed)
        if len(secondary_markdown) > len(primary_markdown) * 1.1:
            return secondary_markdown
        return primary_markdown

    def _disagreement_to_dict(self, disagreement: Disagreement) -> Dict[str, Any]:
        """Convert Disagreement object to dictionary"""
        return {
            "type": disagreement.type.value,
            "location": disagreement.location,
            "provider1_content": disagreement.provider1_content,
            "provider2_content": disagreement.provider2_content,
            "confidence": disagreement.confidence
        }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about configured providers"""
        return {
            "primary": self.primary_provider.get_model_info(),
            "secondary": self.secondary_provider.get_model_info(),
            "adjudicator": self.adjudicator_provider.get_model_info()
        }
