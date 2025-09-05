# import os
# from mistralai import Mistral
# from dotenv import load_dotenv
# load_dotenv()

# class MistralOCRProvider():
#     def __init__(self):

#         self.api_key = "3caG8GEhUTQZnCOXR7r9IcgrSMccaa6z"
#         self.client = Mistral(api_key=self.api_key)

#     def process_mistral_ocr(self, base64_image: str) -> str:
#         # Use the correct OCR API with image base64 inclusion
#         ocr_response = self.client.ocr.process(
#             model="mistral-ocr-latest",
#             document={
#                 "type": "image_url",
#                 "image_url": f"data:image/jpeg;base64,{base64_image}"
#             },
#             include_image_base64=True,
#             # Add other potentially useful parameters
#             image_limit=10,  # Limit number of images extracted
#             image_min_size=100  # Minimum image size
#         )

#         # Log the full response to see what Mistral is returning
#         print(f"Full Mistral OCR response: {ocr_response}")
#         print(f"Response type: {type(ocr_response)}")
#         print(f"Response attributes: {dir(ocr_response)}")

#         print(f"type of pages: {type(ocr_response)}")

#         print(ocr_response.model_dump_json())
#         return ocr_response.pages[0].markdown


import base64
from io import BytesIO
from PIL import Image
import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class MistralOCRProvider():
    def __init__(self):
        self.api_key = "3caG8GEhUTQZnCOXR7r9IcgrSMccaa6z"
        self.client = Mistral(api_key=self.api_key)

    def process_mistral_ocr(self, base64_image: str) -> dict:
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            },
            include_image_base64=True,
            image_limit=10,
            image_min_size=100
        )

        results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results.txt")
        with open(results_path, "w") as f:
            f.write(ocr_response.pages[0].markdown)

        images = self.extract_images(ocr_response)

        saved_files = []
        if images:
            saved_files = self.save_images(images)
            print(f"Saved {len(saved_files)} images")

        print(f"Returning {len(saved_files)} images")
        return saved_files

    def extract_images(self, ocr_response):
        """Extract images from the OCR response"""
        images = []

        response_dict = ocr_response.model_dump()

        # Method 1: Check if images are in pages
        for page_idx, page in enumerate(ocr_response.pages):
            page_dict = page.model_dump() if hasattr(page, 'model_dump') else page

            # Look for image fields in the page
            if 'images' in page_dict:
                for img_idx, img in enumerate(page_dict['images']):
                    if isinstance(img, dict) and 'base64' in img:
                        images.append({
                            'page': page_idx,
                            'index': img_idx,
                            'base64': img['base64']
                        })
                    else:
                        images.append({
                            'page': page_idx,
                            'index': img_idx,
                            'data': img
                        })

            # Check for base64 image data directly
            if 'image_base64' in page_dict:
                images.append({
                    'page': page_idx,
                    'index': 0,
                    'base64': page_dict['image_base64']
                })

        # Method 2: Check top-level response for images
        if 'images' in response_dict:
            for img_idx, img in enumerate(response_dict['images']):
                images.append({
                    'index': img_idx,
                    'data': img
                })

        return images

    def save_images(self, images, output_dir="extracted_images"):
        """Save extracted images to files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for i, img_data in enumerate(images):
            try:
                # Handle different image data formats
                if 'base64' in img_data:
                    base64_data = img_data['base64']
                    # Remove data URL prefix if present
                    if base64_data.startswith('data:'):
                        base64_data = base64_data.split(',', 1)[1]

                    # Decode and save
                    image_bytes = base64.b64decode(base64_data)
                    filename = f"extracted_image_{i}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    saved_files.append(filepath)

                elif 'data' in img_data and isinstance(img_data['data'], dict):
                    # If image data is in a nested structure
                    if 'image_base64' in img_data['data']:
                        base64_data = img_data['data']['image_base64']
                        # Remove data URL prefix if present
                        if base64_data.startswith('data:'):
                            base64_data = base64_data.split(',', 1)[1]

                        # Decode and save
                        image_bytes = base64.b64decode(base64_data)
                        filename = f"extracted_image_{i}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
                        saved_files.append(filepath)
                    elif 'base64' in img_data['data']:
                        base64_data = img_data['data']['base64']
                        # Remove data URL prefix if present
                        if base64_data.startswith('data:'):
                            base64_data = base64_data.split(',', 1)[1]

                        # Decode and save
                        image_bytes = base64.b64decode(base64_data)
                        filename = f"extracted_image_{i}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
                        saved_files.append(filepath)

                elif 'data' in img_data and hasattr(img_data['data'], 'read'):
                    # Handle BytesIO objects
                    bytesio_obj = img_data['data']
                    bytesio_obj.seek(0)  # Reset to beginning
                    image_bytes = bytesio_obj.read()

                    filename = f"extracted_image_{i}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'wb') as f:
                        f.write(image_bytes)
                    saved_files.append(filepath)

                else:
                    continue

            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue

        return saved_files

    def create_image_context_mapping(self, markdown: str, image_files: list) -> dict:
        """Create mapping between images and their surrounding context from markdown"""
        image_mapping = {}
        
        # Parse markdown to find image references and their context
        lines = markdown.split('\n')
        
        for i, image_file in enumerate(image_files):
            # Look for image reference in markdown (e.g., ![img-0.jpeg](img-0.jpeg))
            image_name = f"img-{i}.jpeg"  # Mistral typically names images like this
            context_lines = []
            
            # Find the line with this image reference
            image_line_idx = None
            for line_idx, line in enumerate(lines):
                if image_name in line or f"![img-{i}" in line:
                    image_line_idx = line_idx
                    break
            
            if image_line_idx is not None:
                # Get context around the image (2 lines before and after)
                start_idx = max(0, image_line_idx - 2)
                end_idx = min(len(lines), image_line_idx + 3)
                context_lines = lines[start_idx:end_idx]
            else:
                # If we can't find specific image reference, use section-based context
                # This is a fallback approach
                section_size = len(lines) // max(len(image_files), 1)
                start_idx = i * section_size
                end_idx = min(len(lines), (i + 1) * section_size)
                context_lines = lines[start_idx:end_idx]
            
            # Clean up context (remove empty lines, image references)
            cleaned_context = []
            for line in context_lines:
                line = line.strip()
                if line and not line.startswith('![') and not line.startswith('(img-'):
                    cleaned_context.append(line)
            
            image_mapping[image_file] = {
                "context": '\n'.join(cleaned_context),
                "image_reference": image_name,
                "position_in_document": i
            }
        
        return image_mapping

