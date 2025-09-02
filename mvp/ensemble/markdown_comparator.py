import re
from dataclasses import dataclass
from typing import List, Tuple, Set
from enum import Enum
from difflib import SequenceMatcher


class DisagreementType(Enum):
    NUMBERS = "numbers"
    TABLES = "tables"
    TEXT_CONTENT = "text_content"
    STRUCTURE = "structure"


@dataclass
class Disagreement:
    type: DisagreementType
    location: str
    provider1_content: str
    provider2_content: str
    confidence: float


class MarkdownComparator:
    """Simplified comparator that focuses only on critical disagreements"""

    def __init__(self, disagreement_threshold: float = 0.9):
        self.disagreement_threshold = disagreement_threshold
        # SIMPLIFIED: Much more conservative thresholds - only trigger on huge differences
        self.critical_number_threshold = 0.5   # Only for major number differences (50%+ difference)
        self.critical_field_threshold = 0.3    # Only for very obvious field differences

    def detect_disagreements(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Ultra-simplified disagreement detection - focus only on critical financial data"""
        disagreements = []

        # Only check for critical number disagreements in key financial contexts
        disagreements.extend(self._compare_critical_numbers_only(markdown1, markdown2))
        
        return disagreements
    
    def _contains_numerical_tables(self, markdown1: str, markdown2: str) -> bool:
        """Check if either markdown contains tables with numerical data"""
        for markdown in [markdown1, markdown2]:
            tables = self._extract_tables(markdown)
            for table in tables:
                if re.search(r'\d+\.?\d*', table):  # Contains numbers
                    return True
        return False

    def has_significant_disagreements(self, disagreements: List[Disagreement]) -> bool:
        """SIMPLIFIED: Only trigger for major numerical differences (>$10 or >20% difference)"""
        if not disagreements:
            return False

        for d in disagreements:
            if d.type == DisagreementType.NUMBERS:
                # Check if difference is actually significant
                try:
                    val1 = float(d.provider1_content.replace('$', '').replace(',', ''))
                    val2 = float(d.provider2_content.replace('$', '').replace(',', ''))
                    
                    # Only trigger for major differences: >$10 absolute OR >20% relative
                    abs_diff = abs(val1 - val2)
                    if abs_diff > 10:  # More than $10 difference
                        return True
                    if max(val1, val2) > 0:  # Avoid division by zero
                        rel_diff = abs_diff / max(val1, val2)
                        if rel_diff > 0.2:  # More than 20% difference
                            return True
                except (ValueError, ZeroDivisionError):
                    # If we can't parse numbers, don't trigger adjudication
                    continue
            
        return False
    
    def _get_threshold_for_type(self, disagreement_type: DisagreementType) -> float:
        """Get confidence threshold based on disagreement type"""
        thresholds = {
            DisagreementType.NUMBERS: self.critical_number_threshold,
            DisagreementType.TABLES: self.critical_field_threshold, 
            DisagreementType.TEXT_CONTENT: self.critical_field_threshold,
            DisagreementType.STRUCTURE: self.critical_field_threshold
        }
        return thresholds.get(disagreement_type, self.disagreement_threshold)

    def _compare_critical_numbers_only(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """SIMPLIFIED: Only compare dollar amounts - ignore formatting/context differences"""
        disagreements = []
        
        # SIMPLIFIED: Just look for any dollar amounts anywhere
        dollar_pattern = r'\$([\d,]+(?:\.\d{2})?)'  # Match $123.45 or $1,234
        
        numbers1 = re.findall(dollar_pattern, markdown1)
        numbers2 = re.findall(dollar_pattern, markdown2)
        
        # Convert to floats for comparison
        try:
            values1 = [float(n.replace(',', '')) for n in numbers1]
            values2 = [float(n.replace(',', '')) for n in numbers2]
            
            # Only flag if there are different numbers of dollar amounts or major value differences
            if len(values1) != len(values2):
                # Different count of dollar amounts - potentially significant
                disagreements.append(Disagreement(
                    type=DisagreementType.NUMBERS,
                    location="Dollar amount count",
                    provider1_content=f"{len(values1)} amounts: {numbers1}",
                    provider2_content=f"{len(values2)} amounts: {numbers2}",
                    confidence=0.8
                ))
            else:
                # Same count - check for major value differences
                for i, (v1, v2) in enumerate(zip(values1, values2)):
                    if abs(v1 - v2) > 0.01:  # Any difference at all
                        disagreements.append(Disagreement(
                            type=DisagreementType.NUMBERS,
                            location=f"Dollar amount #{i+1}",
                            provider1_content=numbers1[i] if i < len(numbers1) else "missing",
                            provider2_content=numbers2[i] if i < len(numbers2) else "missing",
                            confidence=0.9
                        ))
        except (ValueError, IndexError):
            # If parsing fails, don't flag disagreements
            pass
        
        return disagreements
    
    def _extract_critical_financial_numbers(self, text: str, patterns: List[str]) -> List[Tuple[str, str]]:
        """Extract only clearly identified financial numbers"""
        results = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    context = match.group(1).strip().lower()
                    value = match.group(2).strip()
                    results.append((context, value))
                elif len(match.groups()) == 1:
                    value = match.group(1).strip()
                    # Look for context around the number
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context_text = text[start:end].lower()
                    if any(word in context_text for word in ['total', 'amount', 'price', 'tax']):
                        results.append(("financial_amount", value))
        
        return results
    
    def _normalize_context(self, context: str) -> str:
        """Normalize context for comparison"""
        context = context.lower().strip()
        # Map similar contexts
        if 'total' in context:
            return 'total'
        elif 'amount' in context:
            return 'amount'
        elif 'tax' in context:
            return 'tax'
        elif 'price' in context:
            return 'price'
        return context

    def _extract_tables(self, text: str) -> List[str]:
        """Extract markdown tables - minimal implementation"""
        tables = []
        lines = text.split('\n')
        current_table = []
        in_table = False

        for line in lines:
            if '|' in line:
                in_table = True
                current_table.append(line)
            elif in_table and current_table:
                # End of table
                tables.append('\n'.join(current_table))
                current_table = []
                in_table = False

        # Don't forget the last table
        if current_table:
            tables.append('\n'.join(current_table))

        return tables