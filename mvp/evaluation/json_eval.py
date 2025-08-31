import re
import json
import editdistance
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple
import os
from datetime import datetime

# ---------- Text Normalization ----------
def normalize_text(text: str) -> str:
    """
    Prepare OCR outputs for fair comparison against ground truth.
    
    Normalization rules:
    1. Remove leading/trailing quotes, escape characters (\n, \t, \r)
    2. Convert all HTML tables to Markdown tables if possible (fallback: strip tags)
    3. Normalize whitespace: collapse multiple spaces, normalize newlines
    4. Lowercase everything
    5. Remove redundant Markdown styling: strip bold (**), italics (*, _), inline code (`...`)
    6. Normalize bullet characters (-, *, +) into "-"
    7. Trim trailing/leading spaces on each line
    """
    if not text:
        return ""
    
    # Step 1: Remove leading/trailing quotes and escape characters
    text = text.strip()
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    
    # Step 2: Simple HTML tag removal
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 3: Normalize whitespace
    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    # Normalize newlines (handle different line endings)
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Step 4: Lowercase everything
    text = text.lower()
    
    # Step 5: Remove redundant Markdown styling
    # Strip bold (**text**)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Strip italics (*text* or _text_)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Strip inline code (`text`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Step 6: Normalize bullet characters
    text = re.sub(r'^[\s]*[\*\+][\s]*', '- ', text, flags=re.MULTILINE)
    
    # Step 7: Trim trailing/leading spaces on each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Final cleanup: remove empty lines and normalize spacing
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple empty lines
    text = text.strip()
    
    return text

# ---------- Data Loading ----------
def load_evaluation_data(data_file: str):
    """Load evaluation data from the collected files."""
    with open(data_file, 'r') as f:
        return json.load(f)

def load_latest_json_data():
    """Load the most recent JSON evaluation data files."""
    # Find the latest timestamp from JSON evaluation files
    files = [f for f in os.listdir('.') if f.startswith('json_evaluation_data_') and f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No JSON evaluation data files found. Please run json_ground_truth.py first.")

    # Get the latest file
    latest_file = sorted(files)[-1]
    timestamp = latest_file.replace('json_evaluation_data_', '').replace('.json', '')

    print(f"Loading JSON data from timestamp: {timestamp}")
    print(f"File: {latest_file}")

    return {
        'json_eval_data': load_evaluation_data(latest_file),
        'timestamp': timestamp
    }

# ---------- JSON Metrics ----------
def cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(ref, hyp) / len(ref)

def json_similarity(ref_json: dict, hyp_json: dict) -> float:
    """Calculate similarity between JSON structures."""
    if not ref_json or not hyp_json:
        return 0.0

    # Convert to strings for comparison
    ref_str = json.dumps(ref_json, sort_keys=True)
    hyp_str = json.dumps(hyp_json, sort_keys=True)

    # Use character-level similarity
    return 1.0 - cer(ref_str, hyp_str)

def semantic_json_similarity(ref_json: dict, hyp_json: dict) -> float:
    """Calculate semantic similarity between JSON structures, ignoring key order."""
    if not ref_json or not hyp_json:
        return 0.0

    def extract_values(obj, path=""):
        """Recursively extract all values with their semantic paths."""
        values = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    values.update(extract_values(value, new_path))
                else:
                    values[new_path] = value
                    
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                if isinstance(item, (dict, list)):
                    values.update(extract_values(item, new_path))
                else:
                    values[new_path] = item
        else:
            values[path] = obj
            
        return values
    
    # Extract all semantic paths and values
    ref_values = extract_values(ref_json)
    hyp_values = extract_values(hyp_json)
    
    if not ref_values:
        return 1.0 if not hyp_values else 0.0
    
    # Calculate value-level similarity
    total_refs = len(ref_values)
    correct_matches = 0
    
    for path, ref_val in ref_values.items():
        if path in hyp_values:
            hyp_val = hyp_values[path]
            
            # Exact match
            if ref_val == hyp_val:
                correct_matches += 1
            # Fuzzy match for strings (handle OCR errors)
            elif isinstance(ref_val, str) and isinstance(hyp_val, str):
                # Use character similarity for string values
                char_sim = 1.0 - cer(str(ref_val).lower(), str(hyp_val).lower())
                if char_sim > 0.8:  # 80% character similarity threshold
                    correct_matches += char_sim
            # Numeric tolerance
            elif isinstance(ref_val, (int, float)) and isinstance(hyp_val, (int, float)):
                if abs(ref_val - hyp_val) <= 0.01:  # Small numeric tolerance
                    correct_matches += 1
    
    # Penalize for extra/missing fields
    extra_fields = len(hyp_values) - len(ref_values)
    penalty = max(0, extra_fields) * 0.1  # 10% penalty per extra field
    
    similarity = correct_matches / total_refs
    return max(0.0, similarity - penalty)

def field_level_accuracy(ref_json: dict, hyp_json: dict) -> Dict[str, float]:
    """Calculate accuracy for each field in the JSON."""
    if not ref_json or not hyp_json:
        return {}
    
    def extract_fields(obj, prefix=""):
        """Extract all fields with their paths."""
        fields = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                field_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    fields.update(extract_fields(value, field_path))
                else:
                    fields[field_path] = value
                    
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                field_path = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    fields.update(extract_fields(item, field_path))
                else:
                    fields[field_path] = item
        else:
            fields[prefix] = obj
            
        return fields
    
    ref_fields = extract_fields(ref_json)
    hyp_fields = extract_fields(hyp_json)
    
    field_accuracies = {}
    
    for field_path, ref_val in ref_fields.items():
        if field_path in hyp_fields:
            hyp_val = hyp_fields[field_path]
            
            # Exact match
            if ref_val == hyp_val:
                field_accuracies[field_path] = 1.0
            # Fuzzy match for strings
            elif isinstance(ref_val, str) and isinstance(hyp_val, str):
                char_sim = 1.0 - cer(str(ref_val).lower(), str(hyp_val).lower())
                field_accuracies[field_path] = char_sim
            # Numeric tolerance
            elif isinstance(ref_val, (int, float)) and isinstance(hyp_val, (int, float)):
                if abs(ref_val - hyp_val) <= 0.01:
                    field_accuracies[field_path] = 1.0
                else:
                    field_accuracies[field_path] = 0.0
            else:
                field_accuracies[field_path] = 0.0
        else:
            field_accuracies[field_path] = 0.0
    
    return field_accuracies

# ---------- JSON Evaluation ----------
def evaluate_json_predictions(json_data: dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate JSON predictions."""
    print("\nEvaluating JSON predictions...")

    records = []
    successful_predictions = 0
    field_accuracies = {}

    for i, (gt, pred, success) in enumerate(zip(json_data['ground_truth'], 
                                               json_data['predictions'], 
                                               json_data['success_flags'])):
        if success and pred:
            # Convert JSON to string for text-based metrics
            gt_str = json.dumps(gt, sort_keys=True)
            pred_str = json.dumps(pred, sort_keys=True)

            # Calculate metrics
            char_error_rate = cer(gt_str, pred_str)
            json_sim = json_similarity(gt, pred)
            semantic_sim = semantic_json_similarity(gt, pred)
            field_acc = field_level_accuracy(gt, pred)
            
            # Aggregate field accuracies
            for field, acc in field_acc.items():
                if field not in field_accuracies:
                    field_accuracies[field] = []
                field_accuracies[field].append(acc)
            
            metrics = {
                'sample_id': json_data['sample_ids'][i],
                'CER': char_error_rate,
                'JSON_Similarity': json_sim,
                'Semantic_JSON_Similarity': semantic_sim,
                'success': True
            }
            
            # Add individual field accuracies
            for field, acc in field_acc.items():
                metrics[f'field_{field}'] = acc
            
            records.append(metrics)
            successful_predictions += 1
        else:
            # Failed prediction
            record = {
                'sample_id': json_data['sample_ids'][i],
                'CER': 1.0,
                'JSON_Similarity': 0.0,
                'Semantic_JSON_Similarity': 0.0,
                'success': False
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Calculate summary metrics
    successful_df = df[df['success'] == True]
    summary = {
        "Total_Samples": len(df),
        "Successful_OCR": successful_predictions,
        "Success_Rate": successful_predictions / len(df) if len(df) > 0 else 0,
        "CER": successful_df["CER"].mean() if len(successful_df) > 0 else 1.0,
        "JSON_Similarity": successful_df["JSON_Similarity"].mean() if len(successful_df) > 0 else 0.0,
        "Semantic_JSON_Similarity": successful_df["Semantic_JSON_Similarity"].mean() if len(successful_df) > 0 else 0.0,
    }
    
    # Add field-level summary
    for field, accuracies in field_accuracies.items():
        if accuracies:
            summary[f'field_{field}_accuracy'] = sum(accuracies) / len(accuracies)

    return df, summary

def save_json_results(df: pd.DataFrame, summary: dict, timestamp: str):
    """Save JSON evaluation results to Excel."""
    filename = f'json_evaluation_results_{timestamp}.xlsx'

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': list(summary.keys()),
            'Value': list(summary.values())
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Detailed results
        df.to_excel(writer, sheet_name='Detailed_Results', index=False)

    print(f"\nJSON results saved to: {filename}")
    return filename

def main():
    """Main JSON evaluation pipeline."""
    print("="*60)
    print("JSON OCR EVALUATION PIPELINE")
    print("="*60)

    try:
        # Load data
        data = load_latest_json_data()

        # Evaluate JSON predictions
        df, summary = evaluate_json_predictions(data['json_eval_data'])

        # Print results
        print("\n" + "="*60)
        print("JSON EVALUATION RESULTS")
        print("="*60)

        print(f"\nJSON OCR Results:")
        print(f"  Total Samples: {summary['Total_Samples']}")
        print(f"  Successful OCR: {summary['Successful_OCR']} ({summary['Success_Rate']*100:.1f}%)")
        print(f"  CER: {summary['CER']:.3f}")
        print(f"  JSON Similarity: {summary['JSON_Similarity']:.3f}")
        print(f"  Semantic JSON Similarity: {summary['Semantic_JSON_Similarity']:.3f}")
        
        # Print field-level accuracies
        field_metrics = {k: v for k, v in summary.items() if k.startswith('field_')}
        if field_metrics:
            print(f"\nField-Level Accuracies:")
            for field, accuracy in field_metrics.items():
                field_name = field.replace('field_', '').replace('_accuracy', '')
                print(f"  {field_name}: {accuracy:.3f}")
        
        # Save results
        results_file = save_json_results(df, summary, data['timestamp'])
        
        print(f"\n✅ JSON evaluation completed successfully!")
        print(f"📊 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during JSON evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
