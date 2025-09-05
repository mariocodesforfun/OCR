from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import io
import base64
import json
import numpy as np
import os
import dotenv

dotenv.load_dotenv()

class GeminiSegmentProvider:

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))


    def parse_json(self, json_output: str):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        json_content = json_output
        
        # Look for ```json markers
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                # Find the content after ```json
                remaining_lines = lines[i+1:]
                # Find the closing ``` and take everything before it
                for j, remaining_line in enumerate(remaining_lines):
                    if remaining_line.strip() == "```":
                        json_content = "\n".join(remaining_lines[:j])
                        break
                else:
                    # If no closing ``` found, take all remaining lines
                    json_content = "\n".join(remaining_lines)
                break
        
        # Clean up the JSON content - remove any trailing content that's not JSON
        json_content = json_content.strip()
        
        # Try to find the end of the JSON array/object
        if json_content.startswith('['):
            # Find the matching closing bracket
            bracket_count = 0
            for i, char in enumerate(json_content):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_content = json_content[:i+1]
                        break
        elif json_content.startswith('{'):
            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_content = json_content[:i+1]
                        break
        
        return json_content

    def _extract_first_valid_json(self, json_text: str):
        """Try to extract the first valid JSON object/array from potentially malformed JSON"""
        try:
            # Try to find the first complete JSON array
            start_idx = json_text.find('[')
            if start_idx != -1:
                bracket_count = 0
                for i in range(start_idx, len(json_text)):
                    if json_text[i] == '[':
                        bracket_count += 1
                    elif json_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            partial_json = json_text[start_idx:i+1]
                            try:
                                return json.loads(partial_json)
                            except json.JSONDecodeError:
                                # Try to fix common JSON issues
                                fixed_json = self._fix_malformed_json(partial_json)
                                return json.loads(fixed_json)
            
            # If no array found, try to find individual objects
            items = []
            start_idx = 0
            while True:
                obj_start = json_text.find('{', start_idx)
                if obj_start == -1:
                    break
                
                brace_count = 0
                for i in range(obj_start, len(json_text)):
                    if json_text[i] == '{':
                        brace_count += 1
                    elif json_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                obj_json = json_text[obj_start:i+1]
                                obj = json.loads(obj_json)
                                items.append(obj)
                                start_idx = i + 1
                                break
                            except json.JSONDecodeError:
                                # Try to fix the malformed object
                                try:
                                    fixed_obj_json = self._fix_malformed_json(obj_json)
                                    obj = json.loads(fixed_obj_json)
                                    items.append(obj)
                                    start_idx = i + 1
                                    break
                                except:
                                    start_idx = i + 1
                                    break
                else:
                    break
            
            return items
            
        except Exception as e:
            print(f"Error extracting valid JSON: {e}")
            return []

    def _fix_malformed_json(self, json_text: str):
        """Attempt to fix common JSON malformation issues"""
        import re
        
        # Fix the specific issue we're seeing: mask field with corrupted content
        # Look for patterns like: "mask": "<start_of_mask>_box_2d": [33, 40, 56, 253]"
        # and replace with a placeholder
        
        # Pattern to find corrupted mask fields
        corrupted_mask_pattern = r'"mask":\s*"[^"]*<start_of_mask>[^"]*"'
        
        # Replace corrupted mask fields with a placeholder
        fixed_json = re.sub(corrupted_mask_pattern, '"mask": "data:image/png;base64,placeholder"', json_text)
        
        # Fix other common issues
        # Remove any unescaped quotes within string values
        # This is a simple approach - for more complex cases, we might need a more sophisticated parser
        
        return fixed_json

    def extract_segmentation_masks(self, prompt: str, image_path: str, output_dir: str = None):
        try:
            print(f"Starting segmentation for image: {image_path}")
            
            # Load and resize image
            im = Image.open(image_path)
            im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            print(f"Image loaded and resized to: {im.size}")

            print(f"THE PROMPT I GOT IS: {prompt}")

            complementary_prompt = """
            Output a JSON list of segmentation masks where each entry contains the 2D
            bounding box in the key "box_2d", the segmentation mask in key "mask", and
            the text label in the key "label". Use descriptive labels.
            """

            full_prompt = f"{prompt}\n\n{complementary_prompt}"
            prompt = full_prompt
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # set thinking_budget to 0 for better results in object detection
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
                config=config
            )

            # Parse JSON response
            print(f"Raw response: {response.text[:200]}...")  # Print first 200 chars
            parsed_json = self.parse_json(response.text)
            print(f"Parsed JSON: {parsed_json[:200]}...")  # Print first 200 chars
            
            # Validate and parse JSON
            try:
                items = json.loads(parsed_json)
                if not isinstance(items, list):
                    print(f"Warning: Expected list but got {type(items)}")
                    items = []
                print(f"Found {len(items)} segmentation items")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Problematic JSON content: {parsed_json[:500]}...")
                # Try to fix the malformed JSON first
                try:
                    fixed_json = self._fix_malformed_json(parsed_json)
                    items = json.loads(fixed_json)
                    if not isinstance(items, list):
                        items = []
                    print(f"Fixed JSON and found {len(items)} segmentation items")
                except json.JSONDecodeError as e2:
                    print(f"Still can't parse after fixing: {e2}")
                    # Try to extract just the first valid JSON object/array
                    items = self._extract_first_valid_json(parsed_json)
                    print(f"Extracted {len(items)} items from partial JSON")

            # Set default output directory if not provided
            if output_dir is None:
                # Create output directory in the project root for easier access
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                output_dir = os.path.join(project_root, "segmentation_outputs")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {os.path.abspath(output_dir)}")
            
            # Create ready_to_ship directory under mvp
            mvp_dir = os.path.dirname(os.path.dirname(__file__))
            ready_to_ship_dir = os.path.join(mvp_dir, "ready_to_ship")
            os.makedirs(ready_to_ship_dir, exist_ok=True)
            print(f"Ready to ship directory: {os.path.abspath(ready_to_ship_dir)}")

            # Process each mask
            for i, item in enumerate(items):
                print(f"Processing item {i}: {item}")
                
                # Check if item has required keys
                if "box_2d" not in item:
                    print(f"Item {i} missing 'box_2d' key, skipping")
                    continue
                    
                if "mask" not in item:
                    print(f"Item {i} missing 'mask' key, skipping")
                    continue
                
                # Sanitize the label for use in filenames
                label = item.get("label", f"item_{i}")
                # Replace invalid filename characters with underscores
                import re
                sanitized_label = re.sub(r'[<>:"/\\|?*]', '_', label)
                # Remove multiple consecutive underscores and trim
                sanitized_label = re.sub(r'_+', '_', sanitized_label).strip('_')
                print(f"Sanitized label: '{label}' -> '{sanitized_label}'")
                
                # Get bounding box coordinates
                box = item["box_2d"]
                y0 = int(box[0] / 1000 * im.size[1])
                x0 = int(box[1] / 1000 * im.size[0])
                y1 = int(box[2] / 1000 * im.size[1])
                x1 = int(box[3] / 1000 * im.size[0])

                # Skip invalid boxes
                if y0 >= y1 or x0 >= x1:
                    print(f"Item {i} has invalid box coordinates, skipping")
                    continue

                # Process mask
                png_str = item["mask"]
                if not png_str.startswith("data:image/png;base64,"):
                    print(f"Item {i} mask doesn't start with expected prefix, skipping")
                    continue

                # Remove prefix
                png_str = png_str.removeprefix("data:image/png;base64,")
                mask_data = base64.b64decode(png_str)
                mask = Image.open(io.BytesIO(mask_data))

                # Resize mask to match bounding box
                mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

                # Convert mask to numpy array for processing
                mask_array = np.array(mask)

                # Create overlay for this mask
                overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)

                # Create overlay for the mask
                color = (255, 255, 255, 200)
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        if mask_array[y - y0, x - x0] > 128:  # Threshold for mask
                            overlay_draw.point((x, y), fill=color)

                # Save individual mask and its overlay
                mask_filename = f"{sanitized_label}_{i}_mask.png"
                overlay_filename = f"{sanitized_label}_{i}_overlay.png"
                
                mask_path = os.path.join(output_dir, mask_filename)
                overlay_path = os.path.join(output_dir, overlay_filename)

                mask.save(mask_path)
                print(f"Saved mask: {mask_path}")

                # Create and save overlay
                composite = Image.alpha_composite(im.convert('RGBA'), overlay)
                composite.save(overlay_path)
                print(f"Saved overlay: {overlay_path}")
                
                # Crop the original image using the bounding box
                cropped_image = im.crop((x0, y0, x1, y1))
                cropped_filename = f"{sanitized_label}_{i}_cropped.png"
                cropped_path = os.path.join(ready_to_ship_dir, cropped_filename)
                cropped_image.save(cropped_path)
                print(f"Saved cropped image: {cropped_path}")

            return items  # Return the parsed segmentation items
            
        except Exception as e:
            print(f"Error in extract_segmentation_masks: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


# if __name__ == "__main__":
#   extract_segmentation_masks("path/to/image.png")