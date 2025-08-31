from typing import Dict, Any
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SchemaConverter:
    """Converts Supreme OCR output to match target JSON schema"""

    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

    def convert_to_schema(self, ocr_output: Dict[str, Any], target_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OCR output to match target schema while preserving data accuracy
        """
        conversion_prompt = f"""You are a data structure converter. Your job is to take extracted OCR data and reformat it to match a specific JSON schema EXACTLY.

CRITICAL RULES:
1. Use ONLY the field names specified in the target schema
2. Do not lose any data - map all relevant information to appropriate schema fields
3. If target schema doesn't have a field for some data, include it in the closest matching field
4. Preserve all numerical values exactly as they are
5. Follow the exact data types specified in schema (string, number, array, object)

SOURCE DATA (from Supreme OCR):
{json.dumps(ocr_output, indent=2)}

TARGET SCHEMA (follow this structure EXACTLY):
{json.dumps(target_schema, indent=2)}

MAPPING RULES:
- "restaurant" data should map to "merchant"
- "order.items" should map to "line_items"
- "transaction" data should map to "receipt_details"
- Keep all accurate numerical values (totals, prices, etc.)
- Combine address components into single address string if required

Convert the source data to match the target schema exactly. Return only valid JSON that conforms to the schema."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": conversion_prompt},
                    {"role": "user", "content": "Convert the data to match the target schema exactly."}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            converted_data = json.loads(response.choices[0].message.content)
            return converted_data

        except Exception as e:
            print(f"Schema conversion failed: {e}")
            # Fallback: return original data
            return ocr_output