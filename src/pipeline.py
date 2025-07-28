"""
Main Pipeline Orchestrator

Coordinates the entire OCR → QA → Validation → JSONL pipeline.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

from ingestion import get_pdfs_to_process
from ocr import run_ocr
from postprocess import clean_ocr_output
from qa_chain import make_qa
from validator import validate

logger = logging.getLogger(__name__)


class OCRQAPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, 
                 output_dir: Path = None,
                 model_name: str = "gpt-3.5-turbo",
                 bert_threshold: float = 0.80,
                 save_intermediate: bool = False):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Output directory for results
            model_name: LLM model for QA generation
            bert_threshold: BERTScore threshold for validation
            save_intermediate: Whether to save intermediate OCR results
        """
        self.output_dir = output_dir or Path(__file__).parent.parent / "data" / "processed"
        self.model_name = model_name
        self.bert_threshold = bert_threshold
        self.save_intermediate = save_intermediate
        
        # Ensure output directories exist
        (self.output_dir / "qa").mkdir(parents=True, exist_ok=True)
        if save_intermediate:
            (self.output_dir / "text").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "latex").mkdir(parents=True, exist_ok=True)
    
    def save_intermediate_results(self, pdf_name: str, text: str, latex: str):
        """
        Save intermediate OCR results.
        
        Args:
            pdf_name: Name of source PDF
            text: OCR text output
            latex: OCR LaTeX output
        """
        base_name = Path(pdf_name).stem
        
        # Save text
        if text:
            text_path = self.output_dir / "text" / f"{base_name}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved text to {text_path}")
        
        # Save LaTeX
        if latex:
            latex_path = self.output_dir / "latex" / f"{base_name}.tex"
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex)
            logger.info(f"Saved LaTeX to {latex_path}")
    
    def save_qa_jsonl(self, qa_pairs: List[Dict], pdf_name: str):
        """
        Save QA pairs to JSONL file.
        
        Args:
            qa_pairs: List of validated QA pairs
            pdf_name: Source PDF name
        """
        if not qa_pairs:
            logger.warning(f"No QA pairs to save for {pdf_name}")
            return
        
        base_name = Path(pdf_name).stem
        output_path = self.output_dir / "qa" / f"{base_name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                # Clean up QA dict for output
                output_qa = {
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'latex': qa.get('latex', ''),
                    'valid_sympy': qa.get('valid_sympy', True),
                    'bertscore_f1': qa.get('bertscore_f1', 0.0)
                }
                f.write(json.dumps(output_qa, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
    
    def process_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF through the pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing statistics
        """
        logger.info(f"{'='*50}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"{'='*50}")
        
        stats = {
            'pdf': pdf_path.name,
            'status': 'failed',
            'ocr_text_length': 0,
            'ocr_latex_length': 0,
            'qa_generated': 0,
            'qa_validated': 0,
            'error': None
        }
        
        try:
            # Step 1: OCR
            logger.info("Step 1/4: Running OCR...")
            text, latex = run_ocr(pdf_path)
            stats['ocr_text_length'] = len(text)
            stats['ocr_latex_length'] = len(latex)
            
            if not text and not latex:
                raise ValueError("No content extracted from PDF")
            
            # Step 2: Post-process
            logger.info("Step 2/4: Post-processing OCR output...")
            cleaned_text, cleaned_latex = clean_ocr_output(text, latex)
            
            # Save intermediate results if requested
            if self.save_intermediate:
                self.save_intermediate_results(pdf_path.name, cleaned_text, cleaned_latex)
            
            # Step 3: Generate QA pairs
            logger.info("Step 3/4: Generating QA pairs...")
            qa_pairs = make_qa(cleaned_text, cleaned_latex, model_name=self.model_name)
            stats['qa_generated'] = len(qa_pairs)
            
            if not qa_pairs:
                logger.warning("No QA pairs generated")
                stats['status'] = 'no_qa'
                return stats
            
            # Step 4: Validate QA pairs
            logger.info("Step 4/4: Validating QA pairs...")
            validated_qa = validate(qa_pairs, bert_threshold=self.bert_threshold)
            stats['qa_validated'] = len(validated_qa)
            
            # Save results
            self.save_qa_jsonl(validated_qa, pdf_path.name)
            
            stats['status'] = 'success'
            logger.info(f"✓ Successfully processed {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"✗ Error processing {pdf_path.name}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def run(self, pdf_paths: Optional[List[Path]] = None):
        """
        Run the complete pipeline.
        
        Args:
            pdf_paths: Optional list of specific PDFs to process
        """
        # Get PDFs to process
        if pdf_paths:
            pdfs = pdf_paths
        else:
            pdfs = get_pdfs_to_process()
        
        if not pdfs:
            logger.warning("No PDFs found to process")
            return
        
        logger.info(f"Starting pipeline for {len(pdfs)} PDFs")
        
        # Process each PDF
        all_stats = []
        for pdf in pdfs:
            stats = self.process_pdf(pdf)
            all_stats.append(stats)
        
        # Summary statistics
        self.print_summary(all_stats)
        
        # Save pipeline run summary
        self.save_run_summary(all_stats)
    
    def print_summary(self, stats_list: List[Dict]):
        """Print pipeline run summary."""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        total = len(stats_list)
        successful = sum(1 for s in stats_list if s['status'] == 'success')
        failed = sum(1 for s in stats_list if s['status'] == 'failed')
        no_qa = sum(1 for s in stats_list if s['status'] == 'no_qa')
        
        total_qa_generated = sum(s['qa_generated'] for s in stats_list)
        total_qa_validated = sum(s['qa_validated'] for s in stats_list)
        
        logger.info(f"Total PDFs processed: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"No QA generated: {no_qa}")
        logger.info(f"Total QA pairs generated: {total_qa_generated}")
        logger.info(f"Total QA pairs validated: {total_qa_validated}")
        logger.info(f"Validation rate: {total_qa_validated/total_qa_generated*100:.1f}%" if total_qa_generated > 0 else "N/A")
    
    def save_run_summary(self, stats_list: List[Dict]):
        """Save run summary to JSON."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'bert_threshold': self.bert_threshold,
            'results': stats_list
        }
        
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved run summary to {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OCR to QA Pipeline")
    parser.add_argument(
        '--pdfs', 
        nargs='+', 
        type=Path,
        help='Specific PDF files to process'
    )
    parser.add_argument(
        '--model',
        default='gpt-3.5-turbo',
        help='LLM model for QA generation'
    )
    parser.add_argument(
        '--bert-threshold',
        type=float,
        default=0.80,
        help='BERTScore threshold for validation'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate OCR results'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = OCRQAPipeline(
        output_dir=args.output_dir,
        model_name=args.model,
        bert_threshold=args.bert_threshold,
        save_intermediate=args.save_intermediate
    )
    
    pipeline.run(pdf_paths=args.pdfs)


if __name__ == "__main__":
    main()