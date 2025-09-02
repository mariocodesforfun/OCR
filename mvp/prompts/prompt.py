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