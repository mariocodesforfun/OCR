from typing import List
import fitz
import tempfile
import os
from fastapi import UploadFile

class PDFProcessor:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def pdf_to_images(self, pdf_file: UploadFile) -> List[str]:
        image_paths = []
        try:
            pdf_bytes = pdf_file.file.read()
            return self._pdf_bytes_to_images(pdf_bytes)

        except Exception as e:
            # Clean up any created files on error
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            raise Exception(f"Error converting PDF to images: {str(e)}")

    def _pdf_bytes_to_images(self, pdf_bytes: bytes) -> List[str]:
        image_paths = []
        pdf_document = None

        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Set zoom factor for high resolution
                zoom = self.dpi / 72  # 72 DPI is default
                mat = fitz.Matrix(zoom, zoom)

                # Render page to image
                pix = page.get_pixmap(matrix=mat)

                img_data = pix.tobytes("png")

                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_page_{page_num + 1}.png") as temp_file:
                    temp_file.write(img_data)
                    image_paths.append(temp_file.name)

            return image_paths

        except Exception as e:
            # Clean up any created files on error
            for img_path in image_paths:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            raise Exception(f"Error converting PDF to images: {str(e)}")

        finally:
            if pdf_document:
                pdf_document.close()

    def preprocess_pdf(self, pdf_file: UploadFile) -> bytes:
        pdf_document = None
        processed_pdf = None

        try:
            # Read PDF bytes from uploaded file
            pdf_bytes = pdf_file.file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Create a new PDF for the processed version
            processed_pdf = fitz.open()

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Basic rotation normalization
                if page.rotation != 0:
                    page.set_rotation(0)

                # Remove problematic annotations
                annots = page.annots()
                for annot in annots:
                    if annot.type[1] in ['Highlight', 'Underline', 'StrikeOut']:
                        page.delete_annot(annot)

                # Copy page to new document
                processed_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

            # Return as bytes
            return processed_pdf.tobytes()

        except Exception as e:
            raise Exception(f"Error preprocessing PDF: {str(e)}")

        finally:
            if pdf_document:
                pdf_document.close()
            if processed_pdf:
                processed_pdf.close()








