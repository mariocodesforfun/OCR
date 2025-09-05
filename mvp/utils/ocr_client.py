import base64
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from prompts.prompt import SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT, build_schema_aware_markdown_prompt
from fastapi import UploadFile

class OpenAiOCRProvider:
    """OpenAI OCR client for extracting text from images."""
    def __init__(self):
        self.client = OpenAI()


    def process_openai_ocr(self, image_bytes: bytes, schema: Optional[Dict[str, Any]] = None) -> str:
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"

            # Use schema-aware prompts if schema is provided
            if schema:
                system_prompt, user_prompt = build_schema_aware_markdown_prompt(schema)
            else:
                system_prompt, user_prompt = SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT

            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [ {"type": "text", "text": user_prompt},
                                                {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                    }
                ]
                # no response_format → default is text
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Markdown extraction failed: {str(e)}")

    def analyze_images_with_context(self, image_context_mapping: dict, schema: dict = None) -> dict:
        """Analyze multiple images with their context using existing prompts."""
        analyzed_results = {}

        for image_path, context_info in image_context_mapping.items():
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()

                # Use existing prompts with context enhancement
                if schema:
                    system_prompt, base_user_prompt = build_schema_aware_markdown_prompt(schema)
                else:
                    system_prompt, base_user_prompt = SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT

                # Add context to user prompt
                context = context_info.get("context", "")
                enhanced_user_prompt = base_user_prompt
                if context:
                    enhanced_user_prompt += f"""

DOCUMENT CONTEXT (for reference):
{context}

Focus on extracting content from this specific image."""

                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f"data:image/png;base64,{encoded_image}"

                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": enhanced_user_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]}
                    ]
                )

                analyzed_results[image_path] = response.choices[0].message.content

            except Exception as e:
                print(f"Error analyzing image {image_path}: {str(e)}")
                analyzed_results[image_path] = f"Error analyzing image: {str(e)}"

        return analyzed_results

    def gpt_segmentation_prompt(self, image_path: str) -> str:
        """
        Generate segmentation prompt using GPT-4o for the given image.
        Args:
            image_path: Path to the image file
        Returns:
            Segmentation instruction string
        """
        import base64
        
        # Read the image file and convert to base64
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error reading image file {image_path}: {str(e)}")

        prompt = """
You are an OCR pipeline assistant.
I will provide you with an image containing structured text (table, form, receipt, document, etc.).

Your task:
Analyze the image and determine the optimal HIGH-LEVEL segmentation approach for OCR processing.

Provide a single, concise instruction that describes:
- The primary segmentation unit (e.g., "process as single page", "segment into individual documents", "divide by major sections", "split into columns")
- Basic reading order (top-to-bottom, left-to-right, column-wise, etc.)
- Any critical structural elements to preserve (headers, table structure, page boundaries)

Keep your response to 2-3 sentences maximum. Focus on the macro-level approach, not detailed subdivision.

Format your answer as:
**Segmentation Approach:** [Brief instruction]

## Example Outputs

**For a multi-column document:**
**Segmentation Approach:** Process as single page with column-aware reading order, maintaining logical flow from left column to right column.

**For a case with multiple diagrams/tables/charts/etc.:**
**Segmentation Approach:** process each diagram/table/chart/etc. as a single unit.

**For a complex table:**
**Segmentation Approach:** process the header as a single unit and each row as a single unit.

**For a case with multiple diagrams/tables/charts/etc. in a page:**
**Segmentation Approach:** process each diagram/table/chart/etc. as a single unit.
"""

        image_url = f"data:image/png;base64,{image_base64}"

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}
            ]
        )
        return response.choices[0].message.content
