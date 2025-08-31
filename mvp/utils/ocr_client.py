import base64
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from prompts.prompt import ocr_prompt


class OCRClient:
    """OpenAI OCR client for extracting text from images."""
    def __init__(self):
        self.client = OpenAI()

    def ocr_json(self, image_bytes: bytes, schema: Optional[dict] = None) -> Dict[str, Any]:
        try:
            # Encode image to base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{encoded_image}"

            # Get OCR prompt
            prompt = ocr_prompt(schema)

            # Prepare API call parameters
            api_params = {
                "model": "gpt-4o-2024-08-06",
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all content as JSON following the exact schema provided"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            }

            # Use structured outputs if schema is provided
            if schema:
                api_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ocr_extraction",
                        "strict": True,
                        "schema": schema
                    }
                }
            else:
                api_params["response_format"] = {"type": "json_object"}

            # Make API call
            response = self.client.chat.completions.create(**api_params)
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")