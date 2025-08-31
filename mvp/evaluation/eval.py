import re
import json
import editdistance
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple
import os
from datetime import datetime
import html
from bs4 import BeautifulSoup

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
    
    # Step 2: Convert HTML tables to Markdown tables
    try:
        # Parse HTML and find tables
        soup = BeautifulSoup(text, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            markdown_table = html_table_to_markdown(table)
            table.replace_with(markdown_table)
        
        # Get text content, stripping remaining HTML tags
        text = soup.get_text()
    except:
        # Fallback: simple HTML tag removal
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

def html_table_to_markdown(table) -> str:
    """Convert HTML table to Markdown table format."""
    try:
        rows = []
        headers = []
        
        # Extract headers (th elements)
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            rows.append(headers)
        
        # Extract data rows
        for tr in table.find_all('tr')[1:]:  # Skip header row
            row = [td.get_text(strip=True) for td in tr.find_all('td')]
            if row:  # Only add non-empty rows
                rows.append(row)
        
        if not rows:
            return ""
        
        # Create markdown table
        markdown_lines = []
        
        # Add header row
        markdown_lines.append('| ' + ' | '.join(rows[0]) + ' |')
        
        # Add separator row
        markdown_lines.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
        
        # Add data rows
        for row in rows[1:]:
            markdown_lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(markdown_lines)
    
    except Exception:
        # Fallback: return table text content
        return table.get_text()

# ---------- Data Loading ----------
def load_evaluation_data(data_file: str):
    """Load evaluation data from the collected files."""
    with open(data_file, 'r') as f:
        return json.load(f)

def load_latest_markdown_data():
    """Load the most recent markdown evaluation data files."""
    # Find the latest timestamp from markdown evaluation files
    files = [f for f in os.listdir('.') if f.startswith('markdown_evaluation_data_') and f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No markdown evaluation data files found. Please run ground_truth_prediciton.py first.")

    # Get the latest file
    latest_file = sorted(files)[-1]
    timestamp = latest_file.replace('markdown_evaluation_data_', '').replace('.json', '')

    print(f"Loading Markdown data from timestamp: {timestamp}")
    print(f"File: {latest_file}")

    return {
        'markdown_eval_data': load_evaluation_data(latest_file),
        'timestamp': timestamp
    }

# ---------- Markdown Metrics ----------
def cer(ref: str, hyp: str) -> float:
    """Character Error Rate."""
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(ref, hyp) / len(ref)

def wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)

def word_accuracy(ref: str, hyp: str) -> float:
    """Proportion of correctly matched words."""
    ref_words, hyp_words = ref.lower().split(), hyp.lower().split()
    if not ref_words:
        return 0.0
    ref_counter, hyp_counter = Counter(ref_words), Counter(hyp_words)
    common = sum((ref_counter & hyp_counter).values())
    return common / len(ref_words)

def extract_table(text: str) -> List[List[str]]:
    """Extract Markdown tables as list-of-lists."""
    tables, current = [], []
    for line in text.splitlines():
        if "|" in line:
            row = [c.strip() for c in line.split("|") if c.strip()]
            if row:
                current.append(row)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables

def table_accuracy(ref: str, hyp: str) -> float:
    """Simple cell overlap metric for markdown tables."""
    ref_tables, hyp_tables = extract_table(ref), extract_table(hyp)
    if not ref_tables:
        return 0.0
    ref_cells = [c for t in ref_tables for r in t for c in r]
    hyp_cells = [c for t in hyp_tables for r in t for c in r]
    if not ref_cells:
        return 0.0
    ref_counter, hyp_counter = Counter(ref_cells), Counter(hyp_cells)
    common = sum((ref_counter & hyp_counter).values())
    return common / len(ref_cells)

def table_structure_accuracy(ref: str, hyp: str) -> Dict[str, float]:
    """Calculate table structure accuracy metrics."""
    ref_tables, hyp_tables = extract_table(ref), extract_table(hyp)
    
    if not ref_tables:
        return {'row_accuracy': 0.0, 'column_accuracy': 0.0, 'cell_accuracy': 0.0}
    
    total_rows = sum(len(table) for table in ref_tables)
    total_columns = sum(len(table[0]) if table else 0 for table in ref_tables)
    total_cells = sum(len(table) * len(table[0]) if table and table[0] else 0 for table in ref_tables)
    
    if total_rows == 0 or total_columns == 0:
        return {'row_accuracy': 0.0, 'column_accuracy': 0.0, 'cell_accuracy': 0.0}
    
    # Row accuracy
    matched_rows = 0
    for ref_table in ref_tables:
        for hyp_table in hyp_tables:
            if len(ref_table) == len(hyp_table):
                matched_rows += len(ref_table)
                break
    
    # Column accuracy (assuming first row has headers)
    matched_columns = 0
    for ref_table in ref_tables:
        for hyp_table in hyp_tables:
            if ref_table and hyp_table and len(ref_table[0]) == len(hyp_table[0]):
                matched_columns += len(ref_table[0])
                break
    
    # Cell accuracy
    matched_cells = 0
    for ref_table in ref_tables:
        for hyp_table in hyp_tables:
            if len(ref_table) == len(hyp_table) and ref_table and hyp_table:
                for ref_row, hyp_row in zip(ref_table, hyp_table):
                    if len(ref_row) == len(hyp_row):
                        matched_cells += len(ref_row)
    
    return {
        'row_accuracy': matched_rows / total_rows if total_rows > 0 else 0.0,
        'column_accuracy': matched_columns / total_columns if total_columns > 0 else 0.0,
        'cell_accuracy': matched_cells / total_cells if total_cells > 0 else 0.0
    }

def sequence_similarity(ref: str, hyp: str) -> float:
    """Calculate sequence similarity using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, ref, hyp).ratio()

# ---------- Markdown Evaluation ----------
def evaluate_ocr(ref: str, hyp: str) -> Dict[str, float]:
    """Evaluate OCR performance on text."""
    # Normalize both reference and hypothesis for fair comparison
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)

    # Basic metrics
    metrics = {
        "CER": cer(ref_norm, hyp_norm),
        "WER": wer(ref_norm, hyp_norm),
        "WordAcc": word_accuracy(ref_norm, hyp_norm),
        "TableAcc": table_accuracy(ref_norm, hyp_norm),
        "SequenceSimilarity": sequence_similarity(ref_norm, hyp_norm)
    }
    
    # Table structure metrics
    table_metrics = table_structure_accuracy(ref_norm, hyp_norm)
    metrics.update(table_metrics)
    
    return metrics

def evaluate_markdown_predictions(markdown_data: list) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate Markdown predictions."""
    print("\nEvaluating Markdown predictions...")

    records = []
    successful_predictions = 0

    for entry in markdown_data:
        gt = entry['ground_truth_markdown']
        pred = entry.get('markdown_prediction')
        success = entry.get('ocr_success', False)
        sample_id = entry['sample_id']
        
        if success and pred:
            metrics = evaluate_ocr(gt, pred)
            metrics['sample_id'] = sample_id
            metrics['success'] = True
            records.append(metrics)
            successful_predictions += 1
        else:
            # Failed prediction
            records.append({
                'sample_id': sample_id,
                'CER': 1.0,
                'WER': 1.0,
                'WordAcc': 0.0,
                'TableAcc': 0.0,
                'SequenceSimilarity': 0.0,
                'row_accuracy': 0.0,
                'column_accuracy': 0.0,
                'cell_accuracy': 0.0,
                'success': False
            })
    
    df = pd.DataFrame(records)
    
    # Calculate summary metrics
    successful_df = df[df['success'] == True]
    summary = {
        "Total_Samples": len(df),
        "Successful_OCR": successful_predictions,
        "Success_Rate": successful_predictions / len(df) if len(df) > 0 else 0,
        "CER": successful_df["CER"].mean() if len(successful_df) > 0 else 1.0,
        "WER": successful_df["WER"].mean() if len(successful_df) > 0 else 1.0,
        "WordAcc": successful_df["WordAcc"].mean() if len(successful_df) > 0 else 0.0,
        "TableAcc": successful_df["TableAcc"].mean() if len(successful_df) > 0 else 0.0,
        "SequenceSimilarity": successful_df["SequenceSimilarity"].mean() if len(successful_df) > 0 else 0.0,
        "RowAccuracy": successful_df["row_accuracy"].mean() if len(successful_df) > 0 else 0.0,
        "ColumnAccuracy": successful_df["column_accuracy"].mean() if len(successful_df) > 0 else 0.0,
        "CellAccuracy": successful_df["cell_accuracy"].mean() if len(successful_df) > 0 else 0.0,
    }

    return df, summary

def save_markdown_results(df: pd.DataFrame, summary: dict, timestamp: str):
    """Save markdown evaluation results to Excel."""
    filename = f'markdown_evaluation_results_{timestamp}.xlsx'

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

    print(f"\nMarkdown results saved to: {filename}")
    return filename

def main():
    """Main markdown evaluation pipeline."""
    print("="*60)
    print("MARKDOWN OCR EVALUATION PIPELINE")
    print("="*60)

    try:
        # Load data
        data = load_latest_markdown_data()

        # Evaluate Markdown predictions
        df, summary = evaluate_markdown_predictions(data['markdown_eval_data'])

        # Print results
        print("\n" + "="*60)
        print("MARKDOWN EVALUATION RESULTS")
        print("="*60)

        print(f"\nMarkdown OCR Results:")
        print(f"  Total Samples: {summary['Total_Samples']}")
        print(f"  Successful OCR: {summary['Successful_OCR']} ({summary['Success_Rate']*100:.1f}%)")
        print(f"  CER: {summary['CER']:.3f}")
        print(f"  WER: {summary['WER']:.3f}")
        print(f"  Word Accuracy: {summary['WordAcc']:.3f}")
        print(f"  Table Accuracy: {summary['TableAcc']:.3f}")
        print(f"  Sequence Similarity: {summary['SequenceSimilarity']:.3f}")
        print(f"  Table Structure:")
        print(f"    Row Accuracy: {summary['RowAccuracy']:.3f}")
        print(f"    Column Accuracy: {summary['ColumnAccuracy']:.3f}")
        print(f"    Cell Accuracy: {summary['CellAccuracy']:.3f}")
        
        # Save results
        results_file = save_markdown_results(df, summary, data['timestamp'])
        
        print(f"\n✅ Markdown evaluation completed successfully!")
        print(f"📊 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during markdown evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
