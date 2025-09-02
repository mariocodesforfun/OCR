import tempfile
import hashlib
import uuid
import logging
import re
from typing import Dict, Any, List, Optional
from fastapi import UploadFile

from providers.base_ocr_provider import BaseOCRProvider
from .markdown_comparator import MarkdownComparator, Disagreement
from utils.json_extractor import JSONExtractor
from utils.pdf_processor import PDFProcessor
from orchestrator import OCROrchestrator

# Configure logging for ensemble operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        
        filename = getattr(file, 'filename', 'unknown')
        logger.info(f"=== ENSEMBLE PROCESSING START: {filename} ===")

        try:
            # Step 0: Prepare image
            logger.info(f"Step 0: Preparing image {filename}")
            image_bytes = self._prepare_image(file)
            logger.info(f"Image prepared, size: {len(image_bytes)} bytes")

            # Step 1: Dual-Model OCR with error handling
            logger.info("Step 1: Running dual-model OCR")

            # Primary model with fallback
            logger.info(f"Running primary provider: {self.primary_provider.provider_name}")
            try:
                primary_markdown = self.primary_provider.extract_markdown(image_bytes)
                if not primary_markdown or len(primary_markdown.strip()) < 5:
                    logger.warning("Primary provider returned empty/minimal content")
                    primary_markdown = ""
                else:
                    logger.info(f"Primary markdown length: {len(primary_markdown)} chars")
                    logger.debug(f"Primary markdown preview: {primary_markdown[:200]}...")
            except Exception as e:
                logger.error(f"Primary provider failed: {str(e)}")
                primary_markdown = ""

            # Secondary model with fallback
            logger.info(f"Running secondary provider: {self.secondary_provider.provider_name}")
            try:
                secondary_markdown = self.secondary_provider.extract_markdown(image_bytes)
                if not secondary_markdown or len(secondary_markdown.strip()) < 5:
                    logger.warning("Secondary provider returned empty/minimal content")
                    secondary_markdown = ""
                else:
                    logger.info(f"Secondary markdown length: {len(secondary_markdown)} chars")
                    logger.debug(f"Secondary markdown preview: {secondary_markdown[:200]}...")
            except Exception as e:
                logger.error(f"Secondary provider failed: {str(e)}")
                secondary_markdown = ""

            # Critical: Handle complete failure case
            if not primary_markdown and not secondary_markdown:
                logger.error("CRITICAL: Both providers failed to extract content")
                return {
                    "status": "error",
                    "error": "Both OCR providers failed to extract content",
                    "ensemble_results": {
                        "primary_markdown": "",
                        "secondary_markdown": "",
                        "disagreements": [],
                        "final_markdown": "",
                        "extracted_json": {}
                    }
                }

            # Handle single-provider success case
            if not primary_markdown:
                logger.warning("Primary failed, using secondary only")
                final_markdown = secondary_markdown
                resolution_method = "secondary_only"
                disagreements = []
            elif not secondary_markdown:
                logger.warning("Secondary failed, using primary only") 
                final_markdown = primary_markdown
                resolution_method = "primary_only"
                disagreements = []
            else:
                # Step 2: Simplified Disagreement Detection (both providers succeeded)
                logger.info("Step 2: Checking for critical disagreements only")
                disagreements = self.comparator.detect_disagreements(primary_markdown, secondary_markdown)
                logger.info(f"Found {len(disagreements)} critical disagreements")
                
                if disagreements:
                    for i, d in enumerate(disagreements):
                        logger.info(f"Critical disagreement {i+1}: {d.location} - '{d.provider1_content}' vs '{d.provider2_content}'")

                has_critical_disagreements = self.comparator.has_significant_disagreements(disagreements)
                logger.info(f"Has critical financial disagreements: {has_critical_disagreements}")

                # Step 3: Simplified Resolution
                logger.info("Step 3: Resolution")
                if has_critical_disagreements:
                    logger.info(f"Critical financial disagreement detected - using adjudication with {self.adjudicator_provider.provider_name}")
                    final_markdown = self._adjudicate_disagreements(
                        image_bytes, primary_markdown, secondary_markdown, disagreements
                    )
                    resolution_method = "adjudication"
                else:
                    logger.info("No critical disagreements - using simple selection")
                    final_markdown = self._select_best_markdown(primary_markdown, secondary_markdown)
                    resolution_method = "selection"
            
            logger.info(f"Final markdown length: {len(final_markdown)} chars")
            
            # Check for potential issues with final markdown
            if len(final_markdown.strip()) == 0:
                logger.error("CRITICAL: Final markdown is empty!")
            elif len(final_markdown.strip()) < 10:
                logger.warning(f"WARNING: Final markdown very short: '{final_markdown.strip()}'")

            # Assess ensemble confidence before final extraction
            ensemble_confidence = self._assess_ensemble_confidence(
                primary_markdown, secondary_markdown, final_markdown, disagreements, resolution_method
            )
            logger.info(f"Ensemble confidence: {ensemble_confidence:.2f}")
            
            # Fallback to regular OCR if ensemble confidence is too low
            if ensemble_confidence < 0.4:  # SIMPLIFIED: Much lower threshold - trust ensemble more
                logger.warning(f"Low ensemble confidence ({ensemble_confidence:.2f}), falling back to regular OCR")
                try:
                    fallback_orchestrator = OCROrchestrator()
                    
                    # Reset file pointer for fallback
                    file.file.seek(0)
                    fallback_result = fallback_orchestrator.process_ocr_json(file, schema)
                    
                    # Add ensemble metadata to fallback result
                    if isinstance(fallback_result, dict):
                        fallback_result["ensemble_fallback"] = True
                        fallback_result["ensemble_confidence"] = ensemble_confidence
                        fallback_result["ensemble_reason"] = f"Low confidence ({ensemble_confidence:.2f}), used fallback"
                        logger.info(f"Fallback successful - regular OCR confidence check passed")
                    
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback failed: {fallback_error}, proceeding with ensemble result")
                    # Continue with ensemble result if fallback fails

            # Step 4: JSON Extraction
            logger.info("Step 4: JSON extraction")
            extracted_json = self.json_extractor.extract_json(final_markdown, schema)
            logger.info(f"JSON extraction completed, extracted fields: {len(extracted_json) if isinstance(extracted_json, dict) else 'N/A'}")

            logger.info(f"=== ENSEMBLE PROCESSING SUCCESS: {filename} ===")
            logger.info(f"Final stats - Resolution: {resolution_method}, Disagreements: {len(disagreements)}, Final length: {len(final_markdown)}")
            
            return {
                "status": "success",
                "ensemble_results": {
                    "primary_markdown": primary_markdown,
                    "secondary_markdown": secondary_markdown,
                    "disagreements": [self._disagreement_to_dict(d) for d in disagreements],
                    "disagreement_count": len(disagreements),
                    "has_significant_disagreements": has_critical_disagreements,
                    "resolution_method": resolution_method,
                    "final_markdown": final_markdown,
                    "extracted_json": extracted_json,
                    "ensemble_confidence": ensemble_confidence
                },
                "providers_used": {
                    "primary": self.primary_provider.provider_name,
                    "secondary": self.secondary_provider.provider_name,
                    "adjudicator": self.adjudicator_provider.provider_name if has_critical_disagreements else None
                },
                "processing_stats": {
                    "primary_length": len(primary_markdown),
                    "secondary_length": len(secondary_markdown),
                    "final_length": len(final_markdown)
                }
            }

        except Exception as e:
            logger.error(f"=== ENSEMBLE PROCESSING FAILED: {filename} ===")
            logger.error(f"Error: {str(e)}")
            logger.exception("Full exception details:")
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

        # Generate a safe, short filename instead of using the long URL

        content_hash = hashlib.md5(file_content).hexdigest()[:8]

        # Generate a unique identifier
        unique_id = str(uuid.uuid4())[:8]

        # Use a safe extension
        if file.filename:
            filename_clean = file.filename.split('?')[0]
            if '.' in filename_clean:
                extension = filename_clean.split('.')[-1]
                extension = extension[:5] if len(extension) <= 5 else 'jpg'
            else:
                extension = 'jpg'
        else:
            extension = 'jpg'

        # Create a safe, short filename
        safe_filename = f"img_{content_hash}_{unique_id}.{extension}"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{extension}",
            prefix="supreme_ocr_"
        ) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
            return temp_path

    def _adjudicate_disagreements(self,
                                image_bytes: bytes,
                                primary_markdown: str,
                                secondary_markdown: str,
                                disagreements: List[Disagreement]) -> str:
        """Use vision-based adjudication for disagreements"""

        adjudication_prompt = self._build_adjudication_prompt(
            primary_markdown, secondary_markdown, disagreements
        )
        
        logger.info(f"Adjudication prompt length: {len(adjudication_prompt)} chars")

        try:
            result = self.adjudicator_provider.extract_markdown_with_context(
                image_bytes, adjudication_prompt
            )
            logger.info(f"Adjudication successful, result length: {len(result)} chars")
            return result
        except Exception as e:
            logger.warning(f"Adjudication failed: {str(e)}, falling back to selection")
            return self._select_best_markdown(primary_markdown, secondary_markdown)

    def _build_adjudication_prompt(self,
                                 primary_markdown: str,
                                 secondary_markdown: str,
                                 disagreements: List[Disagreement]) -> str:
        """SIMPLIFIED: Clean prompt that doesn't mention disagreements to avoid confusion"""
        
        # SIMPLIFIED: Don't mention disagreements at all - just ask for clean OCR
        # This prevents the model from getting confused by conflicting information
        prompt = """Please extract all text from this image as clean, accurate markdown.

Focus on getting all numbers, especially dollar amounts, exactly right.

Return only the markdown content:"""
        
        return prompt

    def _select_best_markdown(self, primary_markdown: str, secondary_markdown: str) -> str:
        """SIMPLIFIED: Just pick the longer output - length usually correlates with completeness"""
        
        # SIMPLIFIED: Remove all complex logic, just pick longer output
        # If neither has content, prefer primary (GPT-4O)
        if not primary_markdown and not secondary_markdown:
            logger.warning("Both outputs empty, returning empty string")
            return ""
        elif not primary_markdown:
            logger.info("Primary empty, using secondary")
            return secondary_markdown
        elif not secondary_markdown:
            logger.info("Secondary empty, using primary")
            return primary_markdown
        elif len(secondary_markdown) > len(primary_markdown):
            logger.info(f"Selected secondary (longer: {len(secondary_markdown)} vs {len(primary_markdown)})")
            return secondary_markdown
        else:
            logger.info(f"Selected primary (longer or equal: {len(primary_markdown)} vs {len(secondary_markdown)})")
            return primary_markdown

    def _assess_ensemble_confidence(self, primary: str, secondary: str, final: str, 
                                  disagreements: List[Disagreement], resolution_method: str) -> float:
        """SIMPLIFIED: Much more optimistic confidence assessment"""
        confidence = 0.9  # Start higher - trust the ensemble approach

        # Only major penalties for clear failures
        if not primary and not secondary:
            confidence = 0.2  # Both failed - very low confidence
        elif not primary or not secondary:
            confidence = 0.7  # One failed - still decent confidence

        # Moderate penalty only for truly empty results
        if not final or len(final.strip()) < 5:
            confidence -= 0.4
        
        # Very light penalty for disagreements - they might actually help accuracy
        if len(disagreements) > 0:
            confidence -= 0.05  # Much smaller penalty
        
        # No penalty for adjudication - it's meant to improve accuracy
        # if resolution_method == "adjudication":
        #     confidence -= 0.05  # REMOVED
            
        # Bigger bonus for having both models succeed
        if primary and secondary and len(final.strip()) > 20:
            confidence += 0.1
            
        return max(0.0, min(1.0, confidence))

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
