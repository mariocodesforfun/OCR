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
                # Step 2: Binary Disagreement Detection (both providers succeeded)
                logger.info("Step 2: Detecting disagreements")
                disagreements = self.comparator.detect_disagreements(primary_markdown, secondary_markdown)
                logger.info(f"Found {len(disagreements)} disagreements")
                
                for i, d in enumerate(disagreements):
                    logger.info(f"Disagreement {i+1}: {d.type.value} at {d.location} (confidence: {d.confidence:.2f})")

                has_disagreements = self.comparator.has_significant_disagreements(disagreements)
                logger.info(f"Has significant disagreements: {has_disagreements}")

                # Step 3: Resolution
                logger.info("Step 3: Resolution")
                if has_disagreements:
                    logger.info(f"Using adjudication with {self.adjudicator_provider.provider_name}")
                    final_markdown = self._adjudicate_disagreements(
                        image_bytes, primary_markdown, secondary_markdown, disagreements
                    )
                    resolution_method = "adjudication"
                else:
                    logger.info("Using selection method")
                    final_markdown = self._select_best_markdown(primary_markdown, secondary_markdown)
                    resolution_method = "selection"
            
            logger.info(f"Final markdown length: {len(final_markdown)} chars")
            
            # Check for potential issues with final markdown
            if len(final_markdown.strip()) == 0:
                logger.error("CRITICAL: Final markdown is empty!")
            elif len(final_markdown.strip()) < 10:
                logger.warning(f"WARNING: Final markdown very short: '{final_markdown.strip()}'")

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
        """Build focused adjudication prompt using smart chunking to avoid context overload"""

        # Check if we need chunking based on total content size
        total_chars = len(primary_markdown) + len(secondary_markdown)
        max_context_size = 3000  # Conservative limit for focused attention
        
        if total_chars <= max_context_size:
            return self._build_full_context_prompt(primary_markdown, secondary_markdown, disagreements)
        else:
            return self._build_chunked_prompt(primary_markdown, secondary_markdown, disagreements)
    
    def _build_full_context_prompt(self, primary_markdown: str, secondary_markdown: str, disagreements: List[Disagreement]) -> str:
        """Build prompt with full context for smaller documents"""
        critical_issues = self._build_critical_issues_summary(disagreements)

        prompt = f"""You are an expert OCR adjudicator resolving conflicts between two AI models.

CRITICAL MISSION: Extract the EXACT content from the image with perfect accuracy.

CONFLICT SUMMARY ({len(disagreements)} issues detected):
"""
        for issue in critical_issues:
            prompt += f"⚠️ {issue}\n"

        prompt += f"""
=== ADJUDICATION RULES ===
1. **NUMBERS ARE SACRED**: Every digit, decimal, currency symbol must be EXACTLY as shown
2. **VISUAL VERIFICATION**: When models disagree, trust what you see in the image
3. **COMPLETE ACCURACY**: Missing or incorrect content = failure

=== MODEL OUTPUTS ===

PRIMARY MODEL:
```
{primary_markdown}
```

SECONDARY MODEL:
```
{secondary_markdown}
```

=== TASK ===
Extract the correct markdown. Focus on resolving the conflicts above.
Return ONLY the corrected markdown - no explanations.
"""
        return prompt
        
    def _build_chunked_prompt(self, primary_markdown: str, secondary_markdown: str, disagreements: List[Disagreement]) -> str:
        """Build focused prompt highlighting only conflicted sections and key context"""
        
        # Extract key sections that have conflicts
        conflict_sections = []
        
        for d in disagreements:
            if d.type.value == "numbers":
                # Extract sections around number conflicts with more context
                conflict_sections.extend(self._extract_number_context(primary_markdown, secondary_markdown, d))
            elif d.type.value == "tables":
                # Extract table sections
                conflict_sections.extend(self._extract_table_context(primary_markdown, secondary_markdown))
            elif d.type.value == "text_content" and "field:" in d.location.lower():
                # Extract key-value sections
                conflict_sections.extend(self._extract_field_context(primary_markdown, secondary_markdown, d))
        
        # Build focused prompt with only relevant sections
        prompt = f"""You are an expert OCR adjudicator. The document is large, so I'm showing you ONLY the sections with conflicts.

CRITICAL MISSION: Resolve conflicts in these specific sections by examining the image.

CONFLICTS TO RESOLVE ({len(disagreements)} issues):
"""
        
        critical_issues = self._build_critical_issues_summary(disagreements)
        for issue in critical_issues:
            prompt += f"⚠️ {issue}\n"

        prompt += f"""
=== RULES ===
1. **EXACT ACCURACY**: Numbers, currency, dates must be pixel-perfect
2. **VISUAL TRUTH**: Trust what you see in the image over model outputs
3. **CONTEXT AWARE**: Consider surrounding text for disambiguation

=== CONFLICTED SECTIONS ONLY ===
"""
        
        for i, (section_type, primary_section, secondary_section) in enumerate(conflict_sections[:5]):  # Limit to top 5 conflicts
            prompt += f"""
SECTION {i+1} ({section_type}):
Primary: {primary_section}
Secondary: {secondary_section}
---"""

        prompt += f"""

=== TASK ===
For each section above, provide the CORRECT version as it appears in the image.
Format: SECTION X: [corrected content]

Focus only on the conflicted sections shown above.
"""
        return prompt
        
    def _build_critical_issues_summary(self, disagreements: List[Disagreement]) -> List[str]:
        """Build summary of critical issues for adjudication"""
        critical_issues = []
        for d in disagreements:
            if d.type.value == "numbers":
                critical_issues.append(f"NUMBER CONFLICT: '{d.provider1_content}' vs '{d.provider2_content}' at {d.location}")
            elif d.type.value == "tables":
                critical_issues.append(f"TABLE STRUCTURE: Different layouts detected")
            elif d.type.value == "text_content":
                if "field:" in d.location.lower():
                    critical_issues.append(f"FIELD MISMATCH: {d.location}")
                else:
                    critical_issues.append(f"TEXT CONTENT: Major differences (confidence: {d.confidence:.1%})")
        return critical_issues
        
    def _extract_number_context(self, primary: str, secondary: str, disagreement: Disagreement) -> List[tuple]:
        """Extract context around number conflicts"""
        sections = []
        
        # Find the conflicting numbers in context (50 chars around)
        primary_content = disagreement.provider1_content
        secondary_content = disagreement.provider2_content
        
        # Find position and extract surrounding context
        primary_context = self._find_content_context(primary, primary_content, 50)
        secondary_context = self._find_content_context(secondary, secondary_content, 50)
        
        if primary_context or secondary_context:
            sections.append(("NUMBER", primary_context or primary_content, secondary_context or secondary_content))
            
        return sections
        
    def _extract_table_context(self, primary: str, secondary: str) -> List[tuple]:
        """Extract table sections for comparison"""
        sections = []
        
        # Extract tables from both
        primary_tables = self._extract_tables_simple(primary)
        secondary_tables = self._extract_tables_simple(secondary)
        
        # Compare first differing table
        if primary_tables and secondary_tables:
            sections.append(("TABLE", primary_tables[0][:200], secondary_tables[0][:200]))
        elif primary_tables:
            sections.append(("TABLE", primary_tables[0][:200], "No table found"))
        elif secondary_tables:
            sections.append(("TABLE", "No table found", secondary_tables[0][:200]))
            
        return sections
        
    def _extract_field_context(self, primary: str, secondary: str, disagreement: Disagreement) -> List[tuple]:
        """Extract key-value field context"""
        sections = []
        
        primary_content = disagreement.provider1_content
        secondary_content = disagreement.provider2_content
        
        # Find the field context (full line containing the field)
        primary_context = self._find_line_context(primary, primary_content)
        secondary_context = self._find_line_context(secondary, secondary_content)
        
        sections.append(("FIELD", primary_context, secondary_context))
        return sections
        
    def _find_content_context(self, text: str, content: str, context_chars: int) -> str:
        """Find content in text and return surrounding context"""
        pos = text.find(content)
        if pos == -1:
            return content
            
        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(content) + context_chars)
        return text[start:end].strip()
        
    def _find_line_context(self, text: str, content: str) -> str:
        """Find content and return the full line containing it"""
        lines = text.split('\n')
        for line in lines:
            if content in line:
                return line.strip()
        return content
        
    def _extract_tables_simple(self, text: str) -> List[str]:
        """Simple table extraction for conflict analysis"""
        tables = []
        lines = text.split('\n')
        current_table = []
        
        for line in lines:
            if '|' in line:
                current_table.append(line)
            elif current_table:
                tables.append('\n'.join(current_table))
                current_table = []
                
        if current_table:
            tables.append('\n'.join(current_table))
            
        return tables

    def _select_best_markdown(self, primary_markdown: str, secondary_markdown: str) -> str:
        """Select the better markdown when no significant disagreements exist"""
        
        def calculate_quality_score(markdown: str) -> float:
            score = 0.0
            
            # Critical: Check for empty or minimal content first
            if not markdown or len(markdown.strip()) < 5:
                return -100  # Heavily penalize empty results
            
            # Enhanced structured content detection
            table_score = _calculate_table_score(markdown)
            score += table_score
            
            # Number density and accuracy indicators
            numbers = re.findall(r'\d+\.?\d*', markdown)
            if numbers:
                score += min(len(numbers) * 2, 15)  # Cap at 15 points
                # Bonus for currency and percentages
                if any(c in markdown for c in ['$', '€', '£', '%']):
                    score += 8

            # Text structure and completeness
            lines = [line.strip() for line in markdown.split('\n') if line.strip()]
            score += min(len(lines), 12)  # Multi-line bonus, capped

            # Specific format indicators (key for documents)
            if ':' in markdown:  # Key-value pairs
                score += 5
            if any(word in markdown.lower() for word in ['total', 'date', 'amount', 'invoice', 'receipt']):
                score += 6
            if re.search(r'\d{2}[/-]\d{2}[/-]\d{2,4}', markdown):  # Date patterns
                score += 4
                
            # Length normalization (more sophisticated)
            length_score = min(len(markdown) / 50, 20)  # Better length scaling
            score += length_score
            
            # Penalize obvious OCR errors
            if len(markdown) > 500 and markdown.count('\n') < 3:  # Long single line
                score -= 15
            if len(set(markdown.replace(' ', ''))) < 10:  # Too few unique characters
                score -= 10
            if markdown.count('?') > len(markdown) / 20:  # Too many unknown characters
                score -= 8
                
            return score
        
        def _calculate_table_score(markdown: str) -> float:
            """Calculate score based on table structure quality"""
            table_score = 0
            
            pipe_lines = [line for line in markdown.split('\n') if '|' in line]
            if len(pipe_lines) >= 2:  # At least header and one row
                table_score += 12
                
                # Check for proper table structure
                if any('---' in line for line in pipe_lines):
                    table_score += 5
                    
                # Count columns consistency
                column_counts = [line.count('|') for line in pipe_lines if '|' in line]
                if column_counts and len(set(column_counts)) == 1:  # Consistent columns
                    table_score += 8
                    
            return table_score
        
        primary_score = calculate_quality_score(primary_markdown)
        secondary_score = calculate_quality_score(secondary_markdown)
        
        logger.info(f"Enhanced scoring: Primary={primary_score:.1f} (len={len(primary_markdown)}), Secondary={secondary_score:.1f} (len={len(secondary_markdown)})")
        
        # More sophisticated selection logic
        score_difference = abs(primary_score - secondary_score)
        
        if score_difference < 3:  # Very close scores
            # Use additional tie-breaking criteria
            primary_numbers = len(re.findall(r'\d+\.?\d*', primary_markdown))
            secondary_numbers = len(re.findall(r'\d+\.?\d*', secondary_markdown))
            
            if primary_numbers != secondary_numbers:
                selected = "primary" if primary_numbers > secondary_numbers else "secondary"
                logger.info(f"Tie broken by number count: {selected}")
                return primary_markdown if selected == "primary" else secondary_markdown
            
            # Default to primary if truly tied
            logger.info("True tie, selecting primary (default)")
            return primary_markdown
        
        selected = "secondary" if secondary_score > primary_score else "primary"
        logger.info(f"Selected {selected} based on enhanced quality score (diff: {score_difference:.1f})")
        return secondary_markdown if secondary_score > primary_score else primary_markdown

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
