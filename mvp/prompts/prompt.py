USER_MARKDOWN_PROMPT = """Extract ALL content as valid Markdown. Focus on perfect accuracy:
- Every number must be exact (check each digit)
- Table structures must match original layout exactly
- Include all text, don't skip anything
- Use proper pipe table format for any tables"""

SYSTEM_MARKDOWN_PROMPT = """You are an OCR engine that converts document images to strict Markdown. 

CRITICAL ACCURACY RULES:
1. **Numbers are sacred** - Never change any digit, decimal, currency, or percentage
2. **Tables must be perfect** - Use pipe-delimited Markdown tables, preserve exact structure
3. **Extract ALL visible text** - No content should be missing
4. **No corrections** - Output exactly what you see, don't fix typos or formatting

For tables: | Column 1 | Column 2 |, with alignment rows: | --- | --- |
"""


JSON_EXTRACTION_SYSTEM_PROMPT = """
  Extract data from the following document based on the JSON schema.
  Return null if the document does not contain information relevant to schema.
  Return only the JSON with no explanation text.
"""

def build_schema_aware_markdown_prompt(schema: dict = None) -> tuple[str, str]:
    """Build schema-aware prompts for better markdown extraction"""

    base_system = """You are an OCR engine that converts document images to strict Markdown. 

CRITICAL ACCURACY RULES:
1. **Numbers are sacred** - Never change any digit, decimal, currency, or percentage
2. **Tables must be perfect** - Use pipe-delimited Markdown tables, preserve exact structure
3. **Extract ALL visible text** - No content should be missing
4. **No corrections** - Output exactly what you see, don't fix typos or formatting

For tables: | Column 1 | Column 2 |, with alignment rows: | --- | --- |"""

    base_user = """Extract ALL content as valid Markdown. Focus on perfect accuracy:
- Every number must be exact (check each digit)
- Table structures must match original layout exactly
- Include all text, don't skip anything
- Use proper pipe table format for any tables"""

    if not schema:
        return base_system, base_user
    
    # Add schema-specific guidance
    schema_guidance = []
    
    # Analyze schema for field-specific instructions
    schema_str = str(schema).lower()
    
    if any(field in schema_str for field in ['total', 'amount', 'price', 'cost', 'tax']):
        schema_guidance.append("- Pay EXTRA attention to all monetary values and totals")
        schema_guidance.append("- Include currency symbols and exact decimal places")
    
    if any(field in schema_str for field in ['date', 'time', 'due']):
        schema_guidance.append("- Extract ALL dates in their original format")
        schema_guidance.append("- Include day, month, year as shown")
    
    if any(field in schema_str for field in ['vendor', 'company', 'supplier', 'from', 'to']):
        schema_guidance.append("- Extract company/vendor names with exact spelling")
        schema_guidance.append("- Include full business names and addresses")
    
    if any(field in schema_str for field in ['items', 'products', 'line', 'description']):
        schema_guidance.append("- Extract ALL line items with quantities and individual prices")
        schema_guidance.append("- Preserve item descriptions exactly as written")
    
    if any(field in schema_str for field in ['invoice', 'number', 'id', 'reference']):
        schema_guidance.append("- Extract all reference numbers and IDs exactly")
        schema_guidance.append("- Include invoice/document numbers completely")
    
    if any(field in schema_str for field in ['shifts', 'schedule', 'employees', 'staff', 'roster']):
        schema_guidance.append("- Extract tabular data with EXTREME precision")
        schema_guidance.append("- Pay special attention to alphanumeric employee IDs - every character matters")
        schema_guidance.append("- Preserve exact time formats (HH:MM-HH:MM)")
        schema_guidance.append("- Distinguish carefully between shift types (Morning/Afternoon/Night/Leave)")
        schema_guidance.append("- Maintain exact table structure and alignment")
    
    if schema_guidance:
        enhanced_user = base_user + "\n\nSCHEMA-SPECIFIC FOCUS:\n" + "\n".join(schema_guidance)
        enhanced_system = base_system + f"\n\nIMPORTANT: The extracted markdown will be used to populate these fields: {', '.join(_extract_field_names(schema))}. Pay special attention to accuracy for these data points."
    else:
        enhanced_user = base_user
        enhanced_system = base_system
    
    return enhanced_system, enhanced_user

def _extract_field_names(schema: dict) -> list[str]:
    """Extract field names from schema for prompt context"""
    fields = []
    
    if isinstance(schema, dict):
        if 'properties' in schema:
            fields.extend(schema['properties'].keys())
        else:
            fields.extend([k for k in schema.keys() if k != 'type'])
    
    return fields[:10]  # Limit to avoid prompt bloat