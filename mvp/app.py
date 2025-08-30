from typing import Union
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from orchestrator import OCROrchestrator

# Load environment variables from .env file
load_dotenv()

ocr_orchestrator = OCROrchestrator()

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/ocr-json")
def ocr_json(file: UploadFile = File(...)):
    return ocr_orchestrator.process_ocr_json(file)
