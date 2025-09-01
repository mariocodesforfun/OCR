# OCR Service

A FastAPI-based OCR service that extracts structured data from PDF documents using best Vision model.

## Quick Start

1. **Install dependencies:**
```bash
cd mvp
pip install -r requirements.txt
```

2. **Set up environment:**
Create a `.env` file in the `mvp` directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Run the server:**
```bash
uvicorn app:app --reload
```

4. **Use the API:**
- Health check: `GET http://localhost:8000/`
- OCR endpoint: `POST http://localhost:8000/v1/ocr-md`
- API docs: `http://localhost:8000/docs`

## Example Usage

```bash
curl -X POST "http://localhost:8000/v1/ocr-md" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

## Project Structure

```
ocr_service/
├── mvp/
│   ├── app.py              # FastAPI application
│   ├── requirements.txt    # Dependencies
│   ├── utils/              # OCR and PDF processing
│   └── prompts/            # OCR prompts
└── README.md
```

## Requirements

- Python 3.11+
- OpenAI API key
