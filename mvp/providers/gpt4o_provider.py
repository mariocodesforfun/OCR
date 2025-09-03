import base64
from typing import Dict, Any, Optional
from openai import OpenAI
from .base_ocr_provider import BaseOCRProvider
from prompts.prompt import SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT, build_schema_aware_markdown_prompt


class GPT4OProvider(BaseOCRProvider):
    """GPT-4o OCR provider implementation"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-08-06"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_markdown(self, image_bytes: bytes, schema: Optional[Dict[str, Any]] = None) -> str:
        """Extract markdown from image using GPT-4o, optionally using schema for better extraction"""
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"

            # Use schema-aware prompts if schema is provided
            if schema:
                system_prompt, user_prompt = build_schema_aware_markdown_prompt(schema)
            else:
                system_prompt, user_prompt = SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"GPT-4o markdown extraction failed: {str(e)}")

    def extract_markdown_with_context(self, image_bytes: bytes, context_prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """Extract markdown with additional context for adjudication"""
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"

            # Use schema-aware system prompt if schema is provided
            system_prompt = build_schema_aware_markdown_prompt(schema)[0] if schema else SYSTEM_MARKDOWN_PROMPT

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"GPT-4o context extraction failed: {str(e)}")

    @property
    def provider_name(self) -> str:
        return f"GPT-4o ({self.model})"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model": self.model,
            "capabilities": ["vision", "text", "markdown"],
            "max_tokens": 4096,
            "supports_context": True
        }
