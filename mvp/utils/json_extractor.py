import json
from openai import OpenAI
from prompts.prompt import JSON_EXTRACTION_SYSTEM_PROMPT

class JSONExtractor:
    def __init__(self):
        self.client = OpenAI()

    def extract_json(self, markdown: str, schema: dict) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": JSON_EXTRACTION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Schema: {json.dumps(schema, indent=2)}\n\nDocument:\n{markdown}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise Exception(f"JSON extraction failed: {str(e)}")
