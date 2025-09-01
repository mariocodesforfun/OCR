from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseOCRProvider(ABC):
    """Abstract base class for OCR providers"""

    @abstractmethod
    def extract_markdown(self, image_bytes: bytes) -> str:
        """Extract markdown from image bytes"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging/identification"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and capabilities"""
        pass

    def extract_markdown_with_context(self, image_bytes: bytes, context_prompt: str) -> str:
        """
        Extract markdown with additional context/prompt.
        Default implementation delegates to extract_markdown.
        Providers can override for context-aware extraction.
        """

        return self.extract_markdown(image_bytes)
