def ocr_prompt():
   return """Extract all text and data from this image as structured JSON.

    For tables: Create JSON array with column headers as field names
    For other content: Use appropriate structure (headings, paragraphs, lists)

    RULES:
    - Preserve all numbers exactly as shown
    - Use descriptive field names based on visible headers
    - Convert empty cells to null
    - No hallucination - only extract visible content
    - Return valid JSON only

    IMPORTANT: USE A SIMULAR SHCEMA TO THE ONE PROVIDED IN THE EXAMPLE

    EXAMPLE:
    {"type":"object","properties":{"latestPatent":{"type":"object","properties":{"date":{"type":"string","description":"Month and year of the patent (ex: 2/1975)"}},"description":"Most recent patent from the references cited."},"patentNumber":{"type":"number","description":"Current patent number"},"patentStatus":{"enum":["A","B1","B2","C","D"],"type":"string","description":"Current patent status (e.g., application or granted status). Appears next to the patent number."}}}

    Extract the content"""