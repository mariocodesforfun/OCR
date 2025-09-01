from typing import Union, Dict, Any
from dotenv import load_dotenv
import json

from fastapi import FastAPI, UploadFile, File, Form
from orchestrator import OCROrchestrator

# Load environment variables from .env file
load_dotenv()

ocr_orchestrator = OCROrchestrator()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/ocr-md")
def ocr_md(file: UploadFile = File(...)):
    return ocr_orchestrator.process_ocr_markdown(file)


@app.post("/v1/ocr-json")
def ocr_json(file: UploadFile = File(...), schema: str = Form(...)):
    try:
        # Parse the schema from JSON string
        parsed_schema = json.loads(schema)
        return ocr_orchestrator.process_ocr_json(file, parsed_schema)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON schema: {str(e)}"}
    except Exception as e:
        return {"error": f"OCR processing failed: {str(e)}"}