"""
Content Filtering Module

Filters out non-mathematical content from OCR output to focus on educational material.
"""

import re
from typing import List, Tuple, Dict
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class ContentFilter:
    """Filters and classifies content to extract mathematical material."""
    
    def __init__(self):
        """Initialize content filter with patterns and keywords."""
        # Mathematical keywords (Korean and English)
        self.math_keywords = {
            # English
            'theorem', 'proof', 'lemma', 'corollary', 'definition', 'proposition',
            'equation', 'formula', 'solve', 'calculate', 'derive', 'prove',
            'function', 'variable', 'integral', 'derivative', 'limit', 'matrix',
            'vector', 'polynomial', 'exponential', 'logarithm', 'trigonometric',
            'sin', 'cos', 'tan', 'sqrt', 'sum', 'product', 'factorial',
            # Korean
            '정리', '증명', '보조정리', '따름정리', '정의', '명제',
            '방정식', '공식', '풀이', '계산', '유도', '함수',
            '변수', '적분', '미분', '극한', '행렬', '벡터',
            '다항식', '지수', '로그', '삼각', '문제', '예제'
        }
        
        # Noise patterns to filter out
        self.noise_patterns = [
            # Page numbers and headers
            r'^\s*\d+\s*$',
            r'^page\s+\d+',
            r'^제\s*\d+\s*장',
            r'^chapter\s+\d+',
            # Publishing info
            r'copyright\s*©',
            r'all\s+rights\s+reserved',
            r'isbn[\s:-]*[\d-]+',
            r'출판사|publisher',
            # URLs and emails
            r'https?://\S+',
            r'\S+@\S+\.\S+',
            # Table of contents
            r'목차|contents|index',
            r'\.{3,}\s*\d+',  # ... page number
            # References
            r'참고문헌|references|bibliography',
        ]
        
        # Mathematical expression patterns
        self.math_patterns = [
            r'[a-zA-Z]\s*=\s*[\d\w\+\-\*/\^\(\)]+',  # equations
            r'\d+\s*[\+\-\*/]\s*\d+',  # arithmetic
            r'\\[a-zA-Z]+\{',  # LaTeX commands
            r'\$[^\$]+\$',  # inline math
            r'∫|∑|∏|√|∂|∇|∆',  # math symbols
            r'\([^\)]*\)\s*=',  # function notation
            r'\d+\.\s*[가-힣\w]+\s*[:.]',  # numbered problems (Korean)
            r'\d+\.\s*\w+\s*[:.]',  # numbered problems (English)
        ]
    
    def calculate_math_density(self, text: str) -> float:
        """
        Calculate the density of mathematical content in text.
        
        Args:
            text: Input text
            
        Returns:
            Math density score (0-1)
        """
        if not text.strip():
            return 0.0
        
        # Count mathematical elements
        math_count = 0
        
        # Count keywords
        text_lower = text.lower()
        for keyword in self.math_keywords:
            math_count += text_lower.count(keyword.lower())
        
        # Count mathematical patterns
        for pattern in self.math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            math_count += len(matches)
        
        # Count mathematical symbols
        math_symbols = '+-*/=≠≤≥<>∫∑∏√∂∇∆αβγδεθλμπσφω'
        for symbol in math_symbols:
            math_count += text.count(symbol)
        
        # Calculate density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        density = min(math_count / word_count, 1.0)
        return density
    
    def is_noise(self, text: str) -> bool:
        """
        Check if text is likely noise (headers, footers, etc.).
        
        Args:
            text: Input text
            
        Returns:
            True if text is noise
        """
        text_strip = text.strip()
        
        # Check if too short
        if len(text_strip) < 10:
            return True
        
        # Check noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, text_strip, re.IGNORECASE):
                return True
        
        # Check if mostly numbers (likely page numbers)
        if re.match(r'^[\d\s\-\.]+$', text_strip):
            return True
        
        return False
    
    def classify_content(self, text: str) -> str:
        """
        Classify content type.
        
        Args:
            text: Input text
            
        Returns:
            Content type: 'math', 'noise', or 'other'
        """
        if self.is_noise(text):
            return 'noise'
        
        math_density = self.calculate_math_density(text)
        
        if math_density > 0.1:  # Threshold for mathematical content
            return 'math'
        
        return 'other'
    
    def filter_chunks(self, chunks: List[str]) -> List[str]:
        """
        Filter list of text chunks to keep only mathematical content.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Filtered list containing only mathematical content
        """
        filtered = []
        
        for chunk in chunks:
            classification = self.classify_content(chunk)
            
            if classification == 'math':
                filtered.append(chunk)
                logger.debug(f"Kept math chunk: {chunk[:50]}...")
            elif classification == 'noise':
                logger.debug(f"Filtered noise: {chunk[:50]}...")
            else:
                # For 'other' content, check if it has some math
                if self.calculate_math_density(chunk) > 0.05:
                    filtered.append(chunk)
                    logger.debug(f"Kept borderline chunk: {chunk[:50]}...")
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} mathematical chunks")
        return filtered
    
    def extract_problems(self, text: str) -> List[Dict[str, str]]:
        """
        Extract individual math problems from text.
        
        Args:
            text: Input text
            
        Returns:
            List of problem dictionaries with 'number' and 'content'
        """
        problems = []
        
        # Pattern for numbered problems
        problem_patterns = [
            r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',  # 1. Problem...
            r'문제\s*(\d+)[:.]\s*([^\n]+(?:\n(?!문제)[^\n]+)*)',  # 문제 1: ...
            r'Example\s*(\d+)[:.]\s*([^\n]+(?:\n(?!Example)[^\n]+)*)',  # Example 1: ...
            r'예제\s*(\d+)[:.]\s*([^\n]+(?:\n(?!예제)[^\n]+)*)',  # 예제 1: ...
        ]
        
        for pattern in problem_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                problem_num = match.group(1)
                problem_content = match.group(2).strip()
                
                # Only keep if it has mathematical content
                if self.calculate_math_density(problem_content) > 0.1:
                    problems.append({
                        'number': problem_num,
                        'content': problem_content
                    })
        
        return problems
    
    def smart_split(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Split text intelligently by mathematical units.
        
        Args:
            text: Input text
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # First, try to extract problems
        problems = self.extract_problems(text)
        
        if problems:
            # Group problems into chunks
            chunks = []
            current_chunk = []
            current_size = 0
            
            for problem in problems:
                problem_text = f"Problem {problem['number']}: {problem['content']}"
                problem_size = len(problem_text)
                
                if current_size + problem_size > max_chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [problem_text]
                    current_size = problem_size
                else:
                    current_chunk.append(problem_text)
                    current_size += problem_size
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            return chunks
        
        # Fall back to paragraph-based splitting
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Filter chunks
        return self.filter_chunks(chunks)


def filter_content(text: str) -> str:
    """
    Convenience function to filter content.
    
    Args:
        text: Input text
        
    Returns:
        Filtered text containing mainly mathematical content
    """
    filter_obj = ContentFilter()
    
    # Split into chunks and filter
    chunks = filter_obj.smart_split(text)
    
    # Rejoin filtered chunks
    return '\n\n'.join(chunks)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.DEBUG)
    
    test_text = """
    Page 42
    
    Chapter 3: Quadratic Equations
    
    All rights reserved © 2024 Publisher Inc.
    
    3.1 Definition
    A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0.
    
    Example 1: Solve x² - 5x + 6 = 0
    Solution: Using factorization, (x-2)(x-3) = 0
    Therefore, x = 2 or x = 3
    
    Visit our website at www.example.com
    
    Problem 1. Find the roots of 2x² + 3x - 5 = 0
    Problem 2. Solve: x² + 4x + 4 = 0
    
    Table of Contents........................... 1
    References................................. 150
    """
    
    filtered = filter_content(test_text)
    print("Filtered content:")
    print(filtered)