import json
import requests
import os
import sys
import pandas as pd
from PIL import Image
import io
from dotenv import load_dotenv
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from utils.ocr_client import OCRClient

def get_dataset_samples(offset=0, length=5):
    """Get dataset samples from Hugging Face."""
    url = f"https://datasets-server.huggingface.co/rows?dataset=getomni-ai%2Focr-benchmark&config=default&split=test&offset={offset}&length={length}"
    response = requests.get(url)
    return response.json()

def download_dataset_images():
    """Download images from dataset and return ground truth data."""
    print("Downloading dataset images...")
    data = get_dataset_samples(offset=0, length=5)
    ground_truth_data = []
    for i, row in enumerate(data['rows']):
        sample = row['row']
        image_url = sample['image']['src']
        true_markdown = sample['true_markdown_output']
        metadata = json.loads(sample['metadata'])
        print(f"Downloading image {i+1}: {image_url}")
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            filename = f"test_files/dataset_image_{i}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved: {filename}")
            ground_truth_data.append({
                'sample_id': sample['id'],
                'image_file': filename,
                'true_markdown': true_markdown,
                'metadata': metadata
            })
        except Exception as e:
            print(f"✗ Error downloading image {i}: {e}")
    print(f"Downloaded {len(ground_truth_data)} images")
    return ground_truth_data

def run_markdown_ocr_on_images(ground_truth_data):
    """Run Markdown OCR on all images."""
    print("\nRunning Markdown OCR on images...")
    ocr_client = OCRClient()
    predictions = []

    for i, gt_data in enumerate(ground_truth_data):
        image_file = gt_data['image_file']
        print(f"Processing {i+1}/{len(ground_truth_data)}: {image_file}")

        try:
            with open(image_file, 'rb') as f:
                image_bytes = f.read()

            # Run Markdown OCR
            markdown_result = ocr_client.markdown_openai(image_bytes)

            predictions.append({
                'sample_id': gt_data['sample_id'],
                'image_file': image_file,
                'markdown_prediction': markdown_result,
                'success': True
            })
            print(f"✓ Markdown OCR completed")

        except Exception as e:
            print(f"✗ Markdown OCR failed: {e}")
            predictions.append({
                'sample_id': gt_data['sample_id'],
                'image_file': image_file,
                'markdown_prediction': None,
                'error': str(e),
                'success': False
            })

    return predictions

def save_markdown_ground_truth_predictions(ground_truth_data, predictions):
    """Save Markdown ground truth and prediction data to files."""
    print("\nSaving Markdown ground truth and predictions...")

    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data for saving
    evaluation_data = []
    for gt, pred in zip(ground_truth_data, predictions):
        data_entry = {
            'sample_id': gt['sample_id'],
            'image_file': gt['image_file'],
            'metadata': gt['metadata'],
            'ground_truth_markdown': gt['true_markdown'],
            'ocr_success': pred['success']
        }

        if pred['success']:
            data_entry.update({
                'markdown_prediction': pred['markdown_prediction']
            })
        else:
            data_entry.update({
                'markdown_prediction': None,
                'error': pred.get('error', 'Unknown error')
            })

        evaluation_data.append(data_entry)

    # Save as JSON
    json_filename = f'markdown_evaluation_data_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(evaluation_data, f, indent=2, default=str)

    # Save as Excel
    excel_filename = f'markdown_evaluation_data_{timestamp}.xlsx'

    # Create DataFrames
    df_main = pd.DataFrame([
        {
            'Sample ID': entry['sample_id'],
            'Image File': entry['image_file'],
            'OCR Success': entry['ocr_success'],
            'Error': entry.get('error', ''),
            'Metadata': json.dumps(entry['metadata'])
        }
        for entry in evaluation_data
    ])

    df_markdown = pd.DataFrame([
        {
            'Sample ID': entry['sample_id'],
            'Ground Truth Markdown': entry['ground_truth_markdown'],
            'Markdown Prediction': entry['markdown_prediction'] if entry['markdown_prediction'] else 'FAILED'
        }
        for entry in evaluation_data
    ])

    # Save to Excel
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Overview', index=False)
        df_markdown.to_excel(writer, sheet_name='Markdown_Data', index=False)

    print(f"✓ Markdown data saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  Excel: {excel_filename}")

    return json_filename, excel_filename

def create_markdown_evaluation_ready_data(ground_truth_data, predictions):
    """Create Markdown data structures ready for evaluation."""
    print("\nPreparing Markdown evaluation-ready data...")

    # Structure for Markdown evaluation
    markdown_evaluation_data = {
        'ground_truth': [gt['true_markdown'] for gt in ground_truth_data],
        'predictions': [pred['markdown_prediction'] for pred in predictions],
        'metadata': [gt['metadata'] for gt in ground_truth_data],
        'sample_ids': [gt['sample_id'] for gt in ground_truth_data],
        'success_flags': [pred['success'] for pred in predictions]
    }

    # Save evaluation-ready data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    markdown_eval_filename = f'markdown_evaluation_ready_{timestamp}.json'
    with open(markdown_eval_filename, 'w') as f:
        json.dump(markdown_evaluation_data, f, indent=2, default=str)

    print(f"✓ Markdown evaluation-ready data saved to:")
    print(f"  Markdown evaluation: {markdown_eval_filename}")

    return markdown_eval_filename

def main():
    """Main Markdown data collection pipeline."""
    print("="*60)
    print("MARKDOWN GROUND TRUTH & PREDICTION DATA COLLECTOR")
    print("="*60)

    # Step 1: Download images and get Markdown ground truth
    ground_truth_data = download_dataset_images()

    # Step 2: Run Markdown OCR
    predictions = run_markdown_ocr_on_images(ground_truth_data)

    # Step 3: Save all Markdown data
    json_file, excel_file = save_markdown_ground_truth_predictions(ground_truth_data, predictions)

    # Step 4: Create Markdown evaluation-ready data
    markdown_eval_file = create_markdown_evaluation_ready_data(ground_truth_data, predictions)

    # Summary
    successful_ocr = sum(1 for pred in predictions if pred['success'])
    total_samples = len(predictions)

    print(f"\n" + "="*60)
    print("MARKDOWN DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Successful Markdown OCR: {successful_ocr} ({successful_ocr/total_samples*100:.1f}%)")
    print(f"Failed Markdown OCR: {total_samples - successful_ocr}")
    print(f"\nFiles created:")
    print(f"  📄 Complete Markdown data: {excel_file}")
    print(f"  📄 JSON format: {json_file}")
    print(f"  🔧 Markdown evaluation ready: {markdown_eval_file}")
    print(f"\nYou can now use these files for Markdown evaluation!")

if __name__ == "__main__":
    main()
