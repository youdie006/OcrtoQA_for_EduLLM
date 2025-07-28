"""
Validation Module

Validates QA pairs using SymPy for mathematical correctness and BERTScore for quality.
"""

from typing import List, Dict, Tuple, Optional
import logging
import re
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from bert_score import score as bert_score
import torch

logger = logging.getLogger(__name__)


class QAValidator:
    """Validates QA pairs for mathematical correctness and quality."""
    
    def __init__(self, bert_model: str = "roberta-large", bert_threshold: float = 0.80):
        """
        Initialize validator.
        
        Args:
            bert_model: BERT model for scoring
            bert_threshold: Minimum BERTScore F1 threshold
        """
        self.bert_model = bert_model
        self.bert_threshold = bert_threshold
        
        # SymPy parsing transformations
        self.transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Check if CUDA is available for BERTScore
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def extract_math_expression(self, text: str) -> Optional[str]:
        """
        Extract mathematical expression from text.
        
        Args:
            text: Input text containing math
            
        Returns:
            Extracted mathematical expression or None
        """
        # Look for equations
        patterns = [
            r'=\s*([^\s,\.]+)',           # After equals sign
            r'^([^\s=]+)\s*$',            # Single expression
            r'x\s*=\s*([^\s,\.]+)',       # x = value
            r':\s*([^\s,\.]+)$',          # After colon
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                return match.group(1).strip()
        
        # Try to extract numeric answer
        numeric_pattern = r'-?\d+\.?\d*'
        match = re.search(numeric_pattern, text)
        if match:
            return match.group(0)
        
        return None
    
    def validate_math_sympy(self, question: str, answer: str) -> Tuple[bool, str]:
        """
        Validate mathematical answer using SymPy.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Extract equation from question if present
            eq_match = re.search(r'([^:]+)=([^,\.]*)(?:[,\.]|$)', question)
            if not eq_match:
                # No equation found, try to validate answer format
                expr = self.extract_math_expression(answer)
                if expr:
                    # Try to parse the expression
                    parsed = parse_expr(expr, transformations=self.transformations)
                    return True, "Valid mathematical expression"
                return False, "No mathematical expression found"
            
            left_side = eq_match.group(1).strip()
            right_side = eq_match.group(2).strip()
            
            # Parse expressions
            left_expr = parse_expr(left_side, transformations=self.transformations)
            right_expr = parse_expr(right_side, transformations=self.transformations)
            
            # Create equation
            equation = sp.Eq(left_expr, right_expr)
            
            # Extract answer value
            answer_expr = self.extract_math_expression(answer)
            if not answer_expr:
                return False, "No mathematical expression in answer"
            
            # Parse answer
            answer_parsed = parse_expr(answer_expr, transformations=self.transformations)
            
            # Find variables in equation
            variables = equation.free_symbols
            
            if len(variables) == 1:
                # Single variable equation
                var = list(variables)[0]
                solutions = sp.solve(equation, var)
                
                # Check if answer matches any solution
                for sol in solutions:
                    if sp.simplify(sol - answer_parsed) == 0:
                        return True, f"Correct solution: {var} = {sol}"
                
                return False, f"Answer {answer_expr} does not match solutions: {solutions}"
            
            elif len(variables) == 0:
                # No variables, check if equation is true
                if equation == True:
                    return True, "Equation is valid"
                else:
                    return False, "Equation is false"
            
            else:
                # Multiple variables, can't fully validate
                return True, "Multi-variable expression (partial validation)"
                
        except Exception as e:
            logger.debug(f"SymPy validation error: {e}")
            # If SymPy can't parse, do basic validation
            if re.search(r'-?\d+\.?\d*', answer):
                return True, "Contains numeric value (SymPy parse failed)"
            return False, f"Mathematical validation failed: {str(e)}"
    
    def calculate_bert_score(self, question: str, answer: str) -> float:
        """
        Calculate BERTScore for QA pair quality.
        
        Args:
            question: Question text
            answer: Answer text
            
        Returns:
            BERTScore F1 score
        """
        try:
            # Create reference by combining question context with answer
            reference = f"The answer to '{question}' is {answer}"
            
            # Calculate BERTScore
            P, R, F1 = bert_score(
                [answer],
                [reference],
                model_type=self.bert_model,
                device=self.device,
                verbose=False
            )
            
            return F1.item()
            
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            return 0.0
    
    def validate_qa_pair(self, qa: Dict) -> Dict:
        """
        Validate a single QA pair.
        
        Args:
            qa: QA dictionary with question, answer, latex
            
        Returns:
            Updated QA dictionary with validation results
        """
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        
        # Validate with SymPy
        sympy_valid, sympy_reason = self.validate_math_sympy(question, answer)
        
        # Calculate BERTScore
        bert_f1 = self.calculate_bert_score(question, answer)
        
        # Update QA with validation results
        validated_qa = qa.copy()
        validated_qa['valid_sympy'] = sympy_valid
        validated_qa['sympy_reason'] = sympy_reason
        validated_qa['bertscore_f1'] = round(bert_f1, 3)
        validated_qa['valid'] = sympy_valid and bert_f1 >= self.bert_threshold
        
        return validated_qa
    
    def validate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Validate all QA pairs and filter valid ones.
        
        Args:
            qa_pairs: List of QA dictionaries
            
        Returns:
            List of validated QA dictionaries (only valid ones)
        """
        validated_pairs = []
        
        for i, qa in enumerate(qa_pairs):
            logger.info(f"Validating QA pair {i+1}/{len(qa_pairs)}")
            
            validated_qa = self.validate_qa_pair(qa)
            
            # Only keep valid pairs
            if validated_qa['valid']:
                validated_pairs.append(validated_qa)
                logger.info(f"✓ Valid: {qa['question'][:50]}...")
            else:
                logger.warning(
                    f"✗ Invalid: {qa['question'][:50]}... "
                    f"(SymPy: {validated_qa['valid_sympy']}, "
                    f"BERT F1: {validated_qa['bertscore_f1']})"
                )
        
        logger.info(f"Validated {len(validated_pairs)}/{len(qa_pairs)} QA pairs")
        return validated_pairs


def validate(qa_pairs: List[Dict], bert_threshold: float = 0.80) -> List[Dict]:
    """
    Convenience function to validate QA pairs.
    
    Args:
        qa_pairs: List of QA dictionaries
        bert_threshold: Minimum BERTScore F1 threshold
        
    Returns:
        List of validated QA dictionaries
    """
    validator = QAValidator(bert_threshold=bert_threshold)
    return validator.validate(qa_pairs)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Test QA pairs
    test_qa_pairs = [
        {
            "question": "Solve for x: 2x + 5 = 11",
            "answer": "3",
            "latex": "2x + 5 = 11"
        },
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "latex": ""
        },
        {
            "question": "Solve: x² - 5x + 6 = 0",
            "answer": "x = 2 or x = 3",
            "latex": "x^2 - 5x + 6 = 0"
        },
        {
            "question": "What is the derivative of x²?",
            "answer": "2x",
            "latex": "\\frac{d}{dx}x^2 = 2x"
        },
        {
            "question": "Solve: 2x + 3 = 7",
            "answer": "5",  # Wrong answer for testing
            "latex": ""
        }
    ]
    
    validated = validate(test_qa_pairs)
    print(f"\nValidation complete: {len(validated)}/{len(test_qa_pairs)} passed")
    
    for qa in validated:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Valid: {qa['valid']} (SymPy: {qa['valid_sympy']}, BERT F1: {qa['bertscore_f1']})")