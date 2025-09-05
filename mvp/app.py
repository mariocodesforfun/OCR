from typing import Union, Dict, Any
from dotenv import load_dotenv
import json
import logging

from fastapi import FastAPI, UploadFile, File, Form
from orchestrator import OCROrchestrator
from multi_model_orchestrator import MultiModelOCROrchestrator

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/ocr-md")
def ocr_md(file: UploadFile = File(...)):
    ocr_orchestrator = OCROrchestrator()
    return ocr_orchestrator.process_ocr_markdown(file)


@app.post("/v1/ocr-json-2")
def ocr_json(file: UploadFile = File(...), schema: str = Form(...)):
    # gpt 4o only
    try:
        parsed_schema = json.loads(schema)
        ocr_orchestrator = OCROrchestrator()
        return ocr_orchestrator.process_ocr_json(file, parsed_schema)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON schema: {str(e)}"}
    except Exception as e:
        return {"error": f"OCR processing failed: {str(e)}"}

@app.post("/v1/ocr-json")
def ocr_json_enhanced(file: UploadFile = File(...), schema: str = Form(...)):
    # gpt 4o + mistral ocr
    print("ocr_json_enhanced")
    try:
        parsed_schema = json.loads(schema)
        multi_model_orchestrator = MultiModelOCROrchestrator()
        result = multi_model_orchestrator.get_enhanced_json_ocr_result(file, parsed_schema)
        print(result)
        return result
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON schema: {str(e)}"}
    except Exception as e:
        return {"error": f"Enhanced OCR processing failed: {str(e)}"}
