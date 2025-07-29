"""
OCR Post-processing Module

Cleans and normalizes OCR output, particularly for mathematical content.
"""

import re
from typing import Dict, List, Tuple
import logging
from content_filter import ContentFilter

logger = logging.getLogger(__name__)


class OCRCleaner:
    """Handles post-processing of OCR output."""
    
    def __init__(self):
        """Initialize cleaner with common OCR error patterns."""
        # Common OCR errors in mathematical text
        self.ocr_replacements = {
            # Greek letters
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'epsilon': 'ε', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
            'pi': 'π', 'sigma': 'σ', 'phi': 'φ', 'omega': 'ω',
            
            # Common OCR mistakes
            'x2': 'x²', 'x3': 'x³', 'xn': 'xⁿ',
            '+-': '±', '=/=': '≠', '<=': '≤', '>=': '≥',
            '->': '→', '<->': '↔', '=>': '⇒', '<=>': '⇔',
            
            # Fraction indicators
            ' / ': '/', '\\frac': '\\frac',
            
            # Common symbol mistakes
            'lim_': 'lim', 'sum_': '∑', 'int_': '∫',
            'sqrt': '√', 'infty': '∞', 'partial': '∂',
            
            # Spacing issues
            '  ': ' ', '\n\n\n': '\n\n',
        }
        
        # Mathematical operation patterns
        self.math_patterns = [
            (r'(\d+)\s*\*\s*(\d+)', r'\1 × \2'),  # Multiplication
            (r'(\d+)\s*/\s*(\d+)', r'\1/\2'),      # Division
            (r'(\w)\s*\^\s*(\d+)', r'\1^\2'),      # Exponents
            (r'sqrt\s*\(([^)]+)\)', r'√(\1)'),     # Square roots
        ]
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in mathematical text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Text with common errors fixed
        """
        result = text
        
        # Apply direct replacements
        for old, new in self.ocr_replacements.items():
            result = result.replace(old, new)
        
        # Apply regex patterns
        for pattern, replacement in self.math_patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove excessive blank lines
        result = []
        prev_blank = False
        for line in lines:
            if line == '':
                if not prev_blank:
                    result.append(line)
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False
        
        return '\n'.join(result)
    
    def fix_math_notation(self, text: str) -> str:
        """
        Fix mathematical notation issues.
        
        Args:
            text: Input text
            
        Returns:
            Text with improved math notation
        """
        # Fix common equation formatting
        text = re.sub(r'([a-zA-Z])\s*=\s*', r'\1 = ', text)
        text = re.sub(r'\s*=\s*([a-zA-Z0-9])', r' = \1', text)
        
        # Fix function notation
        text = re.sub(r'f\s*\(\s*x\s*\)', r'f(x)', text)
        text = re.sub(r'([a-zA-Z])\s*\(\s*([a-zA-Z0-9,\s]+)\s*\)', r'\1(\2)', text)
        
        # Fix limit notation
        text = re.sub(r'lim\s+([a-zA-Z])\s*->\s*([a-zA-Z0-9∞]+)', r'lim_{\1→\2}', text)
        
        # Fix integral notation
        text = re.sub(r'∫\s*([a-zA-Z0-9]+)\s+dx', r'∫\1 dx', text)
        
        return text
    
    def extract_equations(self, text: str) -> List[str]:
        """
        Extract standalone equations from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted equations
        """
        equations = []
        
        # Pattern for standalone equations (lines with = sign)
        eq_pattern = r'^[^=]*[a-zA-Z0-9\s\+\-\*/\^\(\)]+\s*=\s*[^=]+$'
        
        for line in text.split('\n'):
            line = line.strip()
            if re.match(eq_pattern, line):
                equations.append(line)
        
        # Also extract LaTeX equations
        latex_patterns = [
            r'\$([^$]+)\$',           # Inline math
            r'\$\$([^$]+)\$\$',       # Display math
            r'\\begin{equation}(.+?)\\end{equation}',  # Equation environment
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches)
        
        return equations
    
    def clean_latex(self, latex: str) -> str:
        """
        Clean and normalize LaTeX code.
        
        Args:
            latex: Raw LaTeX code
            
        Returns:
            Cleaned LaTeX code
        """
        if not latex:
            return ""
        
        # Remove excessive whitespace
        latex = re.sub(r'\s+', ' ', latex)
        
        # Fix common LaTeX issues
        latex = latex.replace('\\\\', '\\')
        latex = re.sub(r'\\([a-zA-Z]+)([a-zA-Z])', r'\\\1 \2', latex)
        
        # Ensure proper spacing around operators
        latex = re.sub(r'([=+\-*/])', r' \1 ', latex)
        latex = re.sub(r'\s+', ' ', latex)
        
        return latex.strip()
    
    def clean(self, text: str, is_latex: bool = False) -> str:
        """
        Main cleaning function.
        
        Args:
            text: Text to clean
            is_latex: Whether the text is LaTeX code
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        if is_latex:
            return self.clean_latex(text)
        
        # Apply cleaning steps
        result = self.fix_common_ocr_errors(text)
        result = self.fix_math_notation(result)
        result = self.normalize_whitespace(result)
        
        return result


def clean(text: str, is_latex: bool = False) -> str:
    """
    Convenience function to clean OCR text.
    
    Args:
        text: Text to clean
        is_latex: Whether the text is LaTeX code
        
    Returns:
        Cleaned text
    """
    cleaner = OCRCleaner()
    return cleaner.clean(text, is_latex)


def clean_ocr_output(text: str, latex: str) -> Tuple[str, str]:
    """
    Clean both text and LaTeX OCR output with content filtering.
    
    Args:
        text: Raw text from OCR
        latex: Raw LaTeX from OCR
        
    Returns:
        Tuple of (cleaned_text, cleaned_latex)
    """
    cleaner = OCRCleaner()
    content_filter = ContentFilter()
    
    # First clean the text
    cleaned_text = cleaner.clean(text, is_latex=False)
    cleaned_latex = cleaner.clean(latex, is_latex=True)
    
    # Extract equations from text if LaTeX is empty
    if not cleaned_latex and cleaned_text:
        equations = cleaner.extract_equations(cleaned_text)
        if equations:
            cleaned_latex = '\n'.join(equations)
    
    # Apply content filtering to remove non-mathematical content
    if cleaned_text:
        # Split into paragraphs and filter
        paragraphs = cleaned_text.split('\n\n')
        filtered_paragraphs = []
        
        for para in paragraphs:
            if content_filter.classify_content(para) != 'noise':
                filtered_paragraphs.append(para)
        
        cleaned_text = '\n\n'.join(filtered_paragraphs)
    
    return cleaned_text, cleaned_latex


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Test text cleaning
    test_text = """
    Find the value of x:
    x2 + 3x - 4 = 0
    
    Using the quadratic formula:
    x = -b +- sqrt(b2 - 4ac) / 2a
    
    where a=1, b=3, c=-4
    """
    
    cleaned = clean(test_text)
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(cleaned)
    
    # Test LaTeX cleaning
    test_latex = "\\frac{-b  \\pm  \\sqrt{b^2  -  4ac}}{2a}"
    cleaned_latex = clean(test_latex, is_latex=True)
    print(f"\nOriginal LaTeX: {test_latex}")
    print(f"Cleaned LaTeX: {cleaned_latex}")