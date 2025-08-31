from typing import Optional
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

@app.post("/v1/ocr-json")
def ocr_json(file: UploadFile = File(...), schema: Optional[str] = Form(None)):
    # Parse schema if provided
    parsed_schema = None
    if schema:
        try:
            parsed_schema = json.loads(schema)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON schema: {str(e)}"}

    return ocr_orchestrator.process_ocr_json(file, parsed_schema)
