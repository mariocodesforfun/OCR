import json
from openai import OpenAI
from prompts.prompt import JSON_EXTRACTION_SYSTEM_PROMPT

class JSONExtractor:
    def __init__(self):
        self.client = OpenAI()

    def extract_json(self, markdown: str, schema: dict) -> dict:
        try:
            # Enhanced extraction with retry logic
            max_retries = 2
            
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {
                                "role": "system",
                                "content": self._build_enhanced_system_prompt(schema)
                            },
                            {
                                "role": "user",
                                "content": f"Schema: {json.dumps(schema, indent=2)}\n\nDocument:\n{markdown}"
                            }
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1  # Lower temperature for more consistent extraction
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    
                    # Validate result against schema
                    if self._validate_extraction(result, schema):
                        return result
                    else:
                        if attempt == max_retries:
                            # Return partial result with warning
                            return {"_extraction_warning": "Validation failed", **result}
                        else:
                            continue  # Retry
                            
                except json.JSONDecodeError as e:
                    if attempt == max_retries:
                        raise Exception(f"JSON parsing failed after {max_retries + 1} attempts: {str(e)}")
                    continue
                    
        except Exception as e:
            # Fallback: return empty structure matching schema
            fallback_result = self._create_fallback_result(schema)
            fallback_result["_extraction_error"] = str(e)
            return fallback_result
    
    def _build_enhanced_system_prompt(self, schema: dict) -> str:
        """Build enhanced system prompt based on schema fields"""
        base_prompt = """You are a highly accurate OCR data extraction specialist.

CRITICAL ACCURACY REQUIREMENTS:
1. **Numbers**: Extract EXACTLY as written - every digit matters
2. **Currency**: Preserve exact amounts including decimals and symbols  
3. **Dates**: Extract in original format - do not reformat
4. **Text**: Preserve exact spelling and capitalization
5. **Missing data**: Use null for missing fields, never guess

EXTRACTION RULES:"""
        
        # Add field-specific guidance based on schema
        field_guidance = []
        if 'total' in str(schema).lower():
            field_guidance.append("- Extract total amounts with exact precision")
        if 'date' in str(schema).lower():
            field_guidance.append("- Preserve original date format")
        if 'vendor' in str(schema).lower() or 'company' in str(schema).lower():
            field_guidance.append("- Extract company names with exact spelling")
        if 'items' in str(schema).lower() or 'products' in str(schema).lower():
            field_guidance.append("- Extract all line items with quantities and prices")
        
        if field_guidance:
            base_prompt += "\n" + "\n".join(field_guidance)
        
        base_prompt += "\n\nReturn ONLY valid JSON matching the provided schema. No explanations."
        return base_prompt
    
    def _validate_extraction(self, result: dict, schema: dict) -> bool:
        """Basic validation of extracted JSON against schema"""
        try:
            # Check if required fields exist (basic validation)
            if isinstance(schema, dict) and 'required' in schema:
                for field in schema.get('required', []):
                    if field not in result or result[field] is None:
                        return False
            
            # Check for obvious extraction errors
            for key, value in result.items():
                if isinstance(value, str):
                    # Check for obvious OCR errors in numeric fields
                    if any(num_word in key.lower() for num_word in ['total', 'amount', 'price', 'cost']):
                        if value and not any(c.isdigit() or c in '.$,' for c in value):
                            return False
            
            return True
        except:
            return False
    
    def _create_fallback_result(self, schema: dict) -> dict:
        """Create empty structure matching schema for fallback"""
        if isinstance(schema, dict) and 'properties' in schema:
            result = {}
            for field_name, field_schema in schema.get('properties', {}).items():
                if isinstance(field_schema, dict):
                    field_type = field_schema.get('type', 'string')
                    if field_type == 'string':
                        result[field_name] = None
                    elif field_type == 'number':
                        result[field_name] = None
                    elif field_type == 'array':
                        result[field_name] = []
                    elif field_type == 'object':
                        result[field_name] = {}
                    else:
                        result[field_name] = None
            return result
        
        return {}
