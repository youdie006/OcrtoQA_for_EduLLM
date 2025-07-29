"""
OCR to QA Pipeline for Educational LLMs

A complete pipeline for converting mathematical PDFs into validated Q&A datasets.
"""

__version__ = "1.0.0"
__author__ = "OCR-QA Pipeline Team"

# Make key functions available at package level
from .ingestion import get_pdfs_to_process
from .ocr import run_ocr
from .postprocess import clean
from .content_filter import filter_content
from .qa_chain import make_qa
from .validator import validate
from .pipeline import OCRQAPipeline

__all__ = [
    "get_pdfs_to_process",
    "run_ocr",
    "clean",
    "filter_content",
    "make_qa",
    "validate",
    "OCRQAPipeline"
]