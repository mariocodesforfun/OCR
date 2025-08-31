from typing import Optional
import json

def ocr_prompt(schema: Optional[dict] = None):
    base_prompt = """Extract all text and data from this image as structured JSON.

    RULES:
    - Preserve all numbers exactly as shown
    - Convert empty cells to null
    - No hallucination - only extract visible content
    - Return valid JSON only"""

    if schema:
        return f"""{base_prompt}

    CRITICAL: You MUST follow the exact schema structure provided below. Do not add, remove, or rename any fields.

    SCHEMA (follow this structure EXACTLY):
    {json.dumps(schema, indent=2)}

    MANDATORY REQUIREMENTS:
    1. Use ONLY the field names specified in the schema - DO NOT use alternative names
    2. Do not add extra fields not in the schema (like "rank", "silver", etc.)
    3. Follow the exact data types specified (string, number, array, object)  
    4. If schema property has a description, follow it precisely
    5. If you cannot extract a required field, use null
    6. Array items must match the schema's "items" structure exactly

    FIELD MAPPING RULES:
    - If schema says "sliver", use "sliver" (not "silver")
    - If schema says "ranking", use "ranking" (not "rank")
    - If schema says "metal_rankings", use "metal_rankings" (not "medal_rankings")
    - For nation descriptions: follow the description exactly - no abbreviations if description says so

    Extract the content using ONLY the schema fields and structure above. Validate your output matches the schema before responding."""
    else:
        return f"""{base_prompt}

    For tables: Create JSON array with column headers as field names
    For other content: Use appropriate structure (headings, paragraphs, lists)
    Use descriptive field names based on visible headers.

    Extract the content"""