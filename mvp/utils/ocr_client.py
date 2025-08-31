import base64
import json
from typing import Dict, Any
from openai import OpenAI
from prompts.prompt import ocr_prompt


class OCRClient:
    """OpenAI OCR client for extracting text from images."""
    def __init__(self):
        self.client = OpenAI()

    def ocr_json(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            # Encode image to base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"

            # Get OCR prompt
            prompt = ocr_prompt()

            # Make API call
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all content as JSON"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")

    def markdown(self, image_bytes: bytes) -> str:
        try:
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"
            prompt = ocr_prompt()
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all content and return it as **Markdown**."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
                # no response_format → default is text
            )
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Markdown extraction failed: {str(e)}")