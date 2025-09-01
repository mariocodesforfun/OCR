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

    def __init__(self, disagreement_threshold: float = 0.7):
        self.disagreement_threshold = disagreement_threshold

    def detect_disagreements(self, markdown1: str, markdown2: str) -> List[Disagreement]:
        """Detect disagreements between two markdown outputs"""
        disagreements = []

        disagreements.extend(self._compare_numbers(markdown1, markdown2))
        disagreements.extend(self._compare_tables(markdown1, markdown2))
        disagreements.extend(self._compare_text_content(markdown1, markdown2))
        disagreements.extend(self._compare_structure(markdown1, markdown2))

        return disagreements

    def has_significant_disagreements(self, disagreements: List[Disagreement]) -> bool:
        """Determine if disagreements require adjudication"""
        if not disagreements:
            return False

        return any(d.confidence > self.disagreement_threshold for d in disagreements)

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
                if similarity < 0.8:  # Significant difference
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

        if similarity < 0.7:  # Significant text differences
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
        """Extract numbers with their surrounding context"""
        numbers_with_context = []

        # Pattern to match numbers with context
        pattern = r'(.{0,20})(\d+\.?\d*)(.{0,20})'

        for match in re.finditer(pattern, text):
            try:
                number = float(match.group(2))
                context = (match.group(1) + match.group(3)).strip()
                numbers_with_context.append((number, context))
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