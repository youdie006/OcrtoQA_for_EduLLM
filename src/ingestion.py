"""
PDF Ingestion Module

Scans the raw_pdfs directory and returns a list of PDF files to process.
"""

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def scan_pdf_directory(pdf_dir: Path = None) -> List[Path]:
    """
    Scan the PDF directory and return all PDF files.
    
    Args:
        pdf_dir: Path to PDF directory. Defaults to data/raw_pdfs/
        
    Returns:
        List of Path objects for each PDF file found
    """
    if pdf_dir is None:
        # Default to data/raw_pdfs relative to project root
        pdf_dir = Path(__file__).parent.parent / "data" / "raw_pdfs"
    
    if not pdf_dir.exists():
        logger.warning(f"PDF directory {pdf_dir} does not exist")
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    return sorted(pdf_files)


def validate_pdf(pdf_path: Path) -> bool:
    """
    Basic validation to check if file is a valid PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if valid PDF, False otherwise
    """
    if not pdf_path.exists():
        return False
    
    # Check file size (not empty)
    if pdf_path.stat().st_size == 0:
        logger.warning(f"PDF file {pdf_path} is empty")
        return False
    
    # Check PDF magic number
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return False


def get_pdfs_to_process() -> List[Path]:
    """
    Get list of valid PDFs to process.
    
    Returns:
        List of validated PDF paths
    """
    all_pdfs = scan_pdf_directory()
    valid_pdfs = [pdf for pdf in all_pdfs if validate_pdf(pdf)]
    
    logger.info(f"Found {len(valid_pdfs)} valid PDFs to process")
    return valid_pdfs


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    pdfs = get_pdfs_to_process()
    for pdf in pdfs:
        print(f"Found PDF: {pdf}")