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
    """Compares two markdown outputs to detect disagreements"""

    def __init__(self, disagreement_threshold: float = 0.65):
        self.disagreement_threshold = disagreement_threshold
        # More sensitive thresholds for different disagreement types
        self.number_threshold = 0.8
        self.table_threshold = 0.7
        self.text_threshold = 0.6
        self.structure_threshold = 0.5

    def detect_disagreements(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Detect disagreements between two markdown outputs with enhanced JSON-critical detection"""
        disagreements = []

        # Enhanced detection methods
        disagreements.extend(self._compare_numbers(markdown1, markdown2))
        disagreements.extend(self._compare_tables(markdown1, markdown2))
        disagreements.extend(self._compare_text_content(markdown1, markdown2))
        disagreements.extend(self._compare_structure(markdown1, markdown2))
        
        # NEW: JSON-critical field detection
        disagreements.extend(self._compare_json_critical_fields(markdown1, markdown2))
        disagreements.extend(self._compare_currency_amounts(markdown1, markdown2))
        disagreements.extend(self._compare_dates(markdown1, markdown2))

        return disagreements

    def has_significant_disagreements(self, disagreements: List[Disagreement]) -> bool:
        """Determine if disagreements require adjudication using type-specific thresholds"""
        if not disagreements:
            return False

        # Use type-specific thresholds for more accurate detection
        for d in disagreements:
            threshold = self._get_threshold_for_type(d.type)
            if d.confidence > threshold:
                return True

        # Also check for multiple moderate disagreements
        moderate_disagreements = [d for d in disagreements if d.confidence > 0.4]
        if len(moderate_disagreements) >= 3:
            return True
            
        return False
    
    def _get_threshold_for_type(self, disagreement_type: DisagreementType) -> float:
        """Get confidence threshold based on disagreement type"""
        thresholds = {
            DisagreementType.NUMBERS: self.number_threshold,
            DisagreementType.TABLES: self.table_threshold, 
            DisagreementType.TEXT_CONTENT: self.text_threshold,
            DisagreementType.STRUCTURE: self.structure_threshold
        }
        return thresholds.get(disagreement_type, self.disagreement_threshold)

    def _compare_numbers(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare numerical values in both markdowns"""
        disagreements = []

        # Extract numbers with context
        numbers1 = self._extract_numbers_with_context(markdown1)
        numbers2 = self._extract_numbers_with_context(markdown2)

        # Find mismatched numbers
        for (num1, context1) in numbers1:
            matched = False
            for (num2, context2) in numbers2:
                if context1 == context2 and num1 == num2:
                    matched = True
                    break

            if not matched:
                # Look for numbers in same context with different values
                for (num2, context2) in numbers2:
                    if context1 == context2 and num1 != num2:
                        disagreements.append(Disagreement(
                            type=DisagreementType.NUMBERS,
                            location=f"Context: {context1}",
                            provider1_content=str(num1),
                            provider2_content=str(num2),
                            confidence=0.9
                        ))
                        break

        return disagreements

    def _compare_tables(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare table structures and content"""
        disagreements = []

        tables1 = self._extract_tables(markdown1)
        tables2 = self._extract_tables(markdown2)

        # Compare table count
        if len(tables1) != len(tables2):
            disagreements.append(Disagreement(
                type=DisagreementType.TABLES,
                location="Table count",
                provider1_content=f"{len(tables1)} tables",
                provider2_content=f"{len(tables2)} tables",
                confidence=0.8
            ))

        # Compare individual tables
        for i, (table1, table2) in enumerate(zip(tables1, tables2)):
            if table1 != table2:
                # Calculate similarity
                similarity = SequenceMatcher(None, table1, table2).ratio()
                if similarity < 0.85:  # More sensitive to table differences
                    disagreements.append(Disagreement(
                        type=DisagreementType.TABLES,
                        location=f"Table {i+1}",
                        provider1_content=table1[:100] + "..." if len(table1) > 100 else table1,
                        provider2_content=table2[:100] + "..." if len(table2) > 100 else table2,
                        confidence=1.0 - similarity
                    ))

        return disagreements

    def _compare_text_content(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare general text content"""
        disagreements = []

        # Remove tables and code blocks for text comparison
        clean1 = self._clean_markdown_for_text_comparison(markdown1)
        clean2 = self._clean_markdown_for_text_comparison(markdown2)

        # Calculate overall similarity
        similarity = SequenceMatcher(None, clean1, clean2).ratio()

        if similarity < 0.7:  # More sensitive to text differences
            disagreements.append(Disagreement(
                type=DisagreementType.TEXT_CONTENT,
                location="Overall content",
                provider1_content=f"Length: {len(clean1)} chars",
                provider2_content=f"Length: {len(clean2)} chars",
                confidence=1.0 - similarity
            ))

        return disagreements

    def _compare_structure(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare markdown structure (headers, lists, etc.)"""
        disagreements = []

        # Compare header structure
        headers1 = self._extract_headers(markdown1)
        headers2 = self._extract_headers(markdown2)

        if headers1 != headers2:
            disagreements.append(Disagreement(
                type=DisagreementType.STRUCTURE,
                location="Header structure",
                provider1_content=str(headers1),
                provider2_content=str(headers2),
                confidence=0.7
            ))

        # Compare list structures
        lists1 = self._extract_lists(markdown1)
        lists2 = self._extract_lists(markdown2)

        if len(lists1) != len(lists2):
            disagreements.append(Disagreement(
                type=DisagreementType.STRUCTURE,
                location="List count",
                provider1_content=f"{len(lists1)} lists",
                provider2_content=f"{len(lists2)} lists",
                confidence=0.6
            ))

        return disagreements

    def _extract_numbers_with_context(self, text: str) -> List[Tuple[float, str]]:
        """Extract numbers with their surrounding context - enhanced for accuracy"""
        numbers_with_context = []

        # Enhanced pattern to match numbers with better context
        patterns = [
            r'(.{0,30})(\d{1,3}(?:,\d{3})*\.\d{2})(.{0,30})',  # Currency format: 1,234.56
            r'(.{0,30})(\d+\.\d{2})(.{0,30})',                 # Decimal: 123.45
            r'(.{0,20})(\d+\.?\d*)(.{0,20})'                  # General numbers
        ]

        processed_positions = set()
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start_pos = match.start(2)
                if start_pos in processed_positions:
                    continue
                    
                try:
                    number_str = match.group(2).replace(',', '')
                    number = float(number_str)
                    context = (match.group(1) + match.group(3)).strip()
                    # Clean context for better matching
                    context = re.sub(r'\s+', ' ', context)
                    numbers_with_context.append((number, context))
                    processed_positions.add(start_pos)
                except ValueError:
                    continue

        return numbers_with_context

    def _extract_tables(self, text: str) -> List[str]:
        """Extract markdown tables"""
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

    def _extract_headers(self, text: str) -> List[str]:
        """Extract markdown headers"""
        headers = []
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                headers.append(line.strip())
        return headers

    def _extract_lists(self, text: str) -> List[str]:
        """Extract markdown lists"""
        lists = []
        current_list = []

        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('-', '*', '+')) or re.match(r'^\d+\.', stripped):
                current_list.append(stripped)
            elif current_list and stripped == '':
                continue  # Empty line in list
            elif current_list:
                # End of list
                lists.append('\n'.join(current_list))
                current_list = []

        if current_list:
            lists.append('\n'.join(current_list))

        return lists

    def _clean_markdown_for_text_comparison(self, text: str) -> str:
        """Remove markdown formatting for text comparison"""
        # Remove tables
        text = re.sub(r'\|.*?\|', '', text)

        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)

        # Remove headers
        text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)

        return text.strip()
        
    def _compare_json_critical_fields(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare fields commonly extracted to JSON with high accuracy requirements"""
        disagreements = []
        
        # Key-value pattern matching for invoice/receipt fields
        kv_patterns = [
            r'(total|amount|subtotal|tax|discount)\s*:?\s*([\d.,]+)',
            r'(date|invoice|receipt)\s*:?\s*([\w\s/-]+)',
            r'(vendor|company|merchant)\s*:?\s*([\w\s&.,-]+)'
        ]
        
        for pattern in kv_patterns:
            matches1 = re.findall(pattern, markdown1, re.IGNORECASE)
            matches2 = re.findall(pattern, markdown2, re.IGNORECASE)
            
            # Compare extracted key-value pairs
            for key1, value1 in matches1:
                found_match = False
                for key2, value2 in matches2:
                    if key1.lower() == key2.lower():
                        if value1.strip() != value2.strip():
                            disagreements.append(Disagreement(
                                type=DisagreementType.TEXT_CONTENT,
                                location=f"JSON field: {key1}",
                                provider1_content=value1.strip(),
                                provider2_content=value2.strip(), 
                                confidence=0.95  # High confidence for field mismatches
                            ))
                        found_match = True
                        break
                        
                if not found_match:
                    # Field exists in one but not the other
                    disagreements.append(Disagreement(
                        type=DisagreementType.TEXT_CONTENT,
                        location=f"Missing field: {key1}",
                        provider1_content=f"{key1}: {value1}",
                        provider2_content="Field not found",
                        confidence=0.85
                    ))
                    
        return disagreements
        
    def _compare_currency_amounts(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Enhanced currency amount comparison with context"""
        disagreements = []
        
        # More sophisticated currency pattern
        currency_pattern = r'([\$€£¥])\s*([\d,]+\.?\d*)|(\b\d+\.\d{2}\b)\s*([\$€£¥])'
        
        amounts1 = re.findall(currency_pattern, markdown1)
        amounts2 = re.findall(currency_pattern, markdown2)
        
        # Normalize amounts for comparison
        def normalize_amount(match):
            if match[0]:  # $X.XX format
                return f"{match[0]}{match[1]}"
            else:  # X.XX$ format
                return f"{match[3]}{match[2]}"
        
        normalized1 = [normalize_amount(m) for m in amounts1]
        normalized2 = [normalize_amount(m) for m in amounts2]
        
        # Compare normalized amounts
        if set(normalized1) != set(normalized2):
            disagreements.append(Disagreement(
                type=DisagreementType.NUMBERS,
                location="Currency amounts",
                provider1_content=str(normalized1),
                provider2_content=str(normalized2),
                confidence=0.9
            ))
            
        return disagreements

    def _compare_dates(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Compare date formats and values"""
        disagreements = []

        # Multiple date patterns
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',      # YYYY/MM/DD
            r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b' # Month DD, YYYY
        ]

        dates1 = set()
        dates2 = set()

        for pattern in date_patterns:
            dates1.update(re.findall(pattern, markdown1))
            dates2.update(re.findall(pattern, markdown2))

        if dates1 != dates2:
            disagreements.append(Disagreement(
                type=DisagreementType.TEXT_CONTENT,
                location="Date values",
                provider1_content=str(sorted(dates1)),
                provider2_content=str(sorted(dates2)),
                confidence=0.85
            ))

        return disagreements