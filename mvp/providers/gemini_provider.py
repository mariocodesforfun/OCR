import base64
from typing import Dict, Any, Optional
from .base_ocr_provider import BaseOCRProvider
from prompts.prompt import SYSTEM_MARKDOWN_PROMPT, USER_MARKDOWN_PROMPT
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiProvider(BaseOCRProvider):
    """Gemini 2.0 Flash OCR provider implementation"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        try:
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required for GeminiProvider. Install with: pip install google-generativeai")

        # Get API key from environment if not provided
        if api_key is None:
            import os
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY environment variable is required for GeminiProvider")

        # Configure the API key
        genai.configure(api_key=api_key)

        self.model_name = model
        try:
            self.model = genai.GenerativeModel(model)
        except Exception as e:
            # Try with a different model if the specified one fails
            print(f"Warning: Could not load model {model}, trying gemini-1.5-flash")
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.model_name = "gemini-1.5-flash"

    def extract_markdown(self, image_bytes: bytes) -> str:
        """Extract markdown from image using Gemini 2.0 Flash"""
        try:
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))

            prompt = f"{SYSTEM_MARKDOWN_PROMPT}\n\n{USER_MARKDOWN_PROMPT}"

            response = self.model.generate_content([
                prompt,
                image
            ])

            if not response.text:
                raise Exception("Gemini returned empty response")

            return response.text

        except Exception as e:
            raise Exception(f"Gemini markdown extraction failed: {str(e)}")

    def extract_markdown_with_context(self, image_bytes: bytes, context_prompt: str) -> str:
        """Extract markdown with additional context for adjudication"""
        try:
            import io
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))

            full_prompt = f"{SYSTEM_MARKDOWN_PROMPT}\n\n{context_prompt}"

            response = self.model.generate_content([
                full_prompt,
                image
            ])

            if not response.text:
                raise Exception("Gemini returned empty response")

            return response.text

        except Exception as e:
            raise Exception(f"Gemini context extraction failed: {str(e)}")

    @property
    def provider_name(self) -> str:
        return f"Gemini ({self.model_name})"

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Google",
            "model": self.model_name,
            "capabilities": ["vision", "text", "markdown"],
            "max_tokens": 8192,
            "supports_context": True
        }