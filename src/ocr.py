"""
OCR Module

Extracts text and LaTeX formulas from PDF pages using Tesseract and MathPix.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import logging
import pdf2image
import pytesseract
import cv2
import numpy as np
import requests
import base64
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing with Tesseract and MathPix."""
    
    def __init__(self, mathpix_app_id: Optional[str] = None, mathpix_app_key: Optional[str] = None):
        """
        Initialize OCR processor.
        
        Args:
            mathpix_app_id: MathPix API app ID (or set MATHPIX_APP_ID env var)
            mathpix_app_key: MathPix API app key (or set MATHPIX_APP_KEY env var)
        """
        self.mathpix_app_id = mathpix_app_id or os.getenv('MATHPIX_APP_ID')
        self.mathpix_app_key = mathpix_app_key or os.getenv('MATHPIX_APP_KEY')
        
        if self.mathpix_app_id and self.mathpix_app_key:
            self.mathpix_enabled = True
            logger.info("MathPix API credentials found")
        else:
            self.mathpix_enabled = False
            logger.warning("MathPix API credentials not found. LaTeX extraction will be limited.")
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion
            
        Returns:
            List of PIL Image objects
        """
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            logger.info(f"Converted {len(images)} pages from {pdf_path}")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        denoised = cv2.fastNlDenoising(gray)
        
        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Configure Tesseract for better math recognition
            custom_config = r'--oem 3 --psm 6 -l eng+kor+equ'
            
            # Extract text
            text = pytesseract.image_to_string(processed, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""
    
    def extract_latex_mathpix(self, image: Image.Image) -> str:
        """
        Extract LaTeX formulas using MathPix API.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted LaTeX formulas
        """
        if not self.mathpix_enabled:
            return ""
        
        try:
            # Convert image to base64
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare API request
            headers = {
                'app_id': self.mathpix_app_id,
                'app_key': self.mathpix_app_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'src': f'data:image/png;base64,{image_base64}',
                'formats': ['latex_simplified', 'text'],
                'math_inline_delimiters': ['$', '$'],
                'math_display_delimiters': ['$$', '$$']
            }
            
            # Make API request
            response = requests.post(
                'https://api.mathpix.com/v3/text',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('latex_simplified', '')
            else:
                logger.error(f"MathPix API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"MathPix extraction error: {e}")
            return ""
    
    def detect_math_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions likely to contain mathematical formulas.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Simple heuristic: look for regions with math symbols
        # This is a placeholder - could be improved with ML models
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        math_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size and aspect ratio typical for formulas
            if w > 50 and h > 20 and w/h > 0.5:
                math_regions.append((x, y, w, h))
        
        return math_regions
    
    def run_ocr(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Run OCR on a PDF file and extract both text and LaTeX.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, extracted_latex)
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        if not images:
            return "", ""
        
        all_text = []
        all_latex = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            
            # Extract text with Tesseract
            text = self.extract_text_tesseract(image)
            all_text.append(f"--- Page {i+1} ---\n{text}")
            
            # Extract LaTeX with MathPix (if enabled)
            if self.mathpix_enabled:
                latex = self.extract_latex_mathpix(image)
                if latex:
                    all_latex.append(f"--- Page {i+1} ---\n{latex}")
        
        combined_text = "\n\n".join(all_text)
        combined_latex = "\n\n".join(all_latex)
        
        return combined_text, combined_latex


def run_ocr(pdf_path: Path) -> Tuple[str, str]:
    """
    Convenience function to run OCR on a PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (extracted_text, extracted_latex)
    """
    processor = OCRProcessor()
    return processor.run_ocr(pdf_path)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample PDF
    test_pdf = Path("data/raw_pdfs/sample.pdf")
    if test_pdf.exists():
        text, latex = run_ocr(test_pdf)
        print(f"Extracted text length: {len(text)}")
        print(f"Extracted LaTeX length: {len(latex)}")
    else:
        print("No test PDF found. Please add a PDF to data/raw_pdfs/")