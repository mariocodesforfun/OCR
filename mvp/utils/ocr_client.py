import base64
import json
from typing import Dict, Any
from openai import OpenAI
from prompts.prompt import system_prompt, user_prompt


class OCRClient:
    """OpenAI OCR client for extracting text from images."""
    def __init__(self):
        self.client = OpenAI()


    def markdown_openai(self, image_bytes: bytes) -> str:
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": [ {"type": "text", "text": user_prompt()},
                                                {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                    }
                ]
                # no response_format → default is text
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Markdown extraction failed: {str(e)}")