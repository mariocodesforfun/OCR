def ocr_prompt():
   return """Extract all text and data from this image as structured JSON.

    For tables: Create JSON array with column headers as field names
    For other content: Use appropriate structure (headings, paragraphs, lists)

    RULES:
    - Preserve all text exactly as shown - for example if there is a noun as rank, do not convert it to its gerund form
    - Preserve all numbers exactly as shown
    - Use descriptive field names based on visible headers
    - Convert empty cells to null
    - No hallucination - only extract visible content
    - Return valid JSON only

    IMPORTANT: USE A SIMULAR SHCEMA TO THE ONE PROVIDED IN THE EXAMPLE

    EXAMPLE:
   {"medal_rankings":[{"gold":2,"total":3,"bronze":0,"nation":"Australia","sliver":1,"ranking":1},{"gold":1,"total":3,"bronze":1,"nation":"Italy","sliver":1,"ranking":2},{"gold":1,"total":2,"bronze":1,"nation":"Germany","sliver":0,"ranking":3},{"gold":1,"total":1,"bronze":0,"nation":"Soviet Union","sliver":0,"ranking":4},{"gold":0,"total":3,"bronze":1,"nation":"Switzerland","sliver":2,"ranking":5},{"gold":0,"total":1,"bronze":0,"nation":"United States","sliver":1,"ranking":6},{"gold":0,"total":1,"bronze":1,"nation":"Great Britain","sliver":0,"ranking":7},{"gold":0,"total":1,"bronze":1,"nation":"France","sliver":0,"ranking":7}]}

    Extract the content"""

def markdown_prompt():
    return """You are an OCR engine. Extract all visible text and data from the image **as valid Markdown**.

RULES:
- Preserve the text exactly as shown (no corrections, no additions).
- Do not hallucinate content that is not visible.
- Use Markdown syntax consistently (headings, bullet points, and tables).
- If the image contains a table, output it as a Markdown table with correct column alignment.
- If the layout is unclear, prefer plain text over inventing structure.
- Do not wrap the result in code fences.

EXAMPLE (table case):
| Rank | Nation            | Gold | Silver | Bronze | Total |
|------|-------------------|------|--------|--------|-------|
| 1    | **Australia** (AUS) | 2    | 1      | 0      | 3     |
| 2    | **Italy** (ITA)     | 1    | 1      | 1      | 3     |
| 3    | **Germany** (EUA)   | 1    | 0      | 1      | 2     |
| 4    | **Soviet Union** (URS) | 1 | 0      | 0      | 1     |

Extract the content now as Markdown."""


def check_your_work_prompt():
    return """You are an OCR engine. Check the content of the image and the extracted Markdown.

RULES:
- Check if the extracted Markdown is valid.
- Check if the extracted Markdown is consistent with the image.
- Check if the extracted Markdown is complete.
- Check if the extracted Markdown is accurate.
- Check if the extracted Markdown is complete.
"""