"""
QA Generation Module

Uses LangChain to generate question-answer pairs from mathematical content.
"""

from typing import List, Dict
import logging
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates QA pairs from mathematical text using LangChain."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.2):
        """
        Initialize QA generator.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. QA generation will fail.")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Create prompts
        self.qa_prompt = PromptTemplate(
            input_variables=["text", "latex"],
            template="""You are a mathematics education expert. Given the following mathematical content, 
generate high-quality question-answer pairs suitable for educational purposes.

Text content:
{text}

LaTeX formulas (if any):
{latex}

Generate question-answer pairs following these guidelines:
1. Questions should test understanding of mathematical concepts
2. Include both conceptual and computational questions
3. Answers should be precise and mathematically correct
4. For computational problems, show the final answer clearly
5. Format each QA pair as JSON with fields: question, answer, latex (if applicable)

Generate 3-5 question-answer pairs. Output only valid JSON array.

Example output:
[
    {{
        "question": "Solve for x: 2x + 5 = 11",
        "answer": "3",
        "latex": "2x + 5 = 11"
    }},
    {{
        "question": "What is the derivative of f(x) = x²?",
        "answer": "f'(x) = 2x",
        "latex": "f(x) = x^2"
    }}
]

QA pairs:"""
        )
        
        self.chunk_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract key mathematical concepts and problems from this text:

{text}

Focus on:
- Mathematical equations and formulas
- Problem statements
- Theorems and definitions
- Solution methods

Summary:"""
        )
    
    def split_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        """
        Split text into manageable chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            
        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def generate_qa_for_chunk(self, text_chunk: str, latex_chunk: str = "") -> List[Dict]:
        """
        Generate QA pairs for a single text chunk.
        
        Args:
            text_chunk: Text content
            latex_chunk: LaTeX content (optional)
            
        Returns:
            List of QA dictionaries
        """
        try:
            # Create chain
            qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
            
            # Generate QA pairs
            result = qa_chain.run(text=text_chunk, latex=latex_chunk)
            
            # Parse JSON result
            qa_pairs = json.loads(result)
            
            # Validate structure
            validated_pairs = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    validated_pairs.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'latex': qa.get('latex', '')
                    })
            
            return validated_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse QA JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}")
            return []
    
    def merge_similar_questions(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or very similar questions.
        
        Args:
            qa_pairs: List of QA pairs
            
        Returns:
            Deduplicated list
        """
        unique_pairs = []
        seen_questions = set()
        
        for qa in qa_pairs:
            # Simple deduplication by lowercase question
            q_lower = qa['question'].lower().strip()
            if q_lower not in seen_questions:
                seen_questions.add(q_lower)
                unique_pairs.append(qa)
        
        return unique_pairs
    
    def make_qa(self, text: str, latex: str = "") -> List[Dict]:
        """
        Generate QA pairs from text and LaTeX content.
        
        Args:
            text: OCR text content
            latex: OCR LaTeX content
            
        Returns:
            List of QA dictionaries
        """
        if not text and not latex:
            logger.warning("No content provided for QA generation")
            return []
        
        # Split text into chunks for processing
        text_chunks = self.split_text(text) if text else [""]
        
        # Split LaTeX similarly if provided
        latex_chunks = self.split_text(latex) if latex else [""] * len(text_chunks)
        
        # Generate QA pairs for each chunk
        all_qa_pairs = []
        for i, (text_chunk, latex_chunk) in enumerate(zip(text_chunks, latex_chunks)):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
            qa_pairs = self.generate_qa_for_chunk(text_chunk, latex_chunk)
            all_qa_pairs.extend(qa_pairs)
        
        # Deduplicate
        unique_pairs = self.merge_similar_questions(all_qa_pairs)
        
        logger.info(f"Generated {len(unique_pairs)} unique QA pairs")
        return unique_pairs


def make_qa(text: str, latex: str = "", model_name: str = "gpt-3.5-turbo") -> List[Dict]:
    """
    Convenience function to generate QA pairs.
    
    Args:
        text: OCR text content
        latex: OCR LaTeX content
        model_name: OpenAI model to use
        
    Returns:
        List of QA dictionaries
    """
    generator = QAGenerator(model_name=model_name)
    return generator.make_qa(text, latex)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Test QA generation
    test_text = """
    Chapter 3: Quadratic Equations
    
    A quadratic equation is a polynomial equation of degree 2. 
    The general form is ax² + bx + c = 0, where a ≠ 0.
    
    Example: Solve x² - 5x + 6 = 0
    Using factoring: (x - 2)(x - 3) = 0
    Therefore, x = 2 or x = 3
    """
    
    test_latex = r"\[x^2 - 5x + 6 = 0\]"
    
    # Note: This will only work if OPENAI_API_KEY is set
    try:
        qa_pairs = make_qa(test_text, test_latex)
        print(f"Generated {len(qa_pairs)} QA pairs:")
        for qa in qa_pairs:
            print(f"\nQ: {qa['question']}")
            print(f"A: {qa['answer']}")
            if qa.get('latex'):
                print(f"LaTeX: {qa['latex']}")
    except Exception as e:
        print(f"Test failed (likely missing API key): {e}")