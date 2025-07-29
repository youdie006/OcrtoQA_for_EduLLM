# Local Math-PDF → OCR → QA Generation → Validation → JSONL Pipeline

A complete pipeline for converting mathematical textbooks and problem books (PDF) into high-quality Question-Answer pairs for educational LLMs.

## 🎯 Key Features

- **100% Local Processing**: No cloud uploads required
- **Math-Focused OCR**: Tesseract for text + MathPix API for LaTeX formulas
- **Smart Content Filtering**: Automatically removes non-mathematical content (headers, footers, etc.)
- **GPT-4o Powered**: Uses latest GPT-4o model for high-quality QA generation
- **Intelligent QA Generation**: LangChain-powered question-answer pair creation
- **Rigorous Validation**: SymPy for mathematical correctness + BERTScore (>0.80) for quality
- **JSONL Output**: Stream-friendly format for large datasets

## 🔗 Pipeline Flow

```
raw_pdfs/*.pdf
    └──➤ ocr.py              # Tesseract (text) + MathPix (LaTeX)
            └──➤ postprocess.py     # OCR cleanup + content filtering
                    └──➤ qa_chain.py       # GPT-4o powered QA generation
                            └──➤ validator.py      # SymPy.equals & BERTScore filter
                                    └──➤ processed/qa/*.jsonl   # final dataset
```

## 🚀 Quick Start

### Prerequisites

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor tesseract-ocr-equ poppler-utils

# macOS
brew install tesseract tesseract-lang poppler
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (create `.env` file):
```bash
OPENAI_API_KEY=your_openai_api_key
MATHPIX_APP_ID=your_mathpix_app_id  # Optional
MATHPIX_APP_KEY=your_mathpix_app_key  # Optional
```

### Running the Pipeline

1. Place PDF files in `data/raw_pdfs/`
2. Run the pipeline:
```bash
python src/pipeline.py
```

### Command Line Options

```bash
# Process specific PDFs
python src/pipeline.py --pdfs path/to/file1.pdf path/to/file2.pdf

# Use different LLM model
python src/pipeline.py --model gpt-4

# Adjust validation threshold
python src/pipeline.py --bert-threshold 0.85

# Save intermediate OCR results
python src/pipeline.py --save-intermediate

# Set custom output directory
python src/pipeline.py --output-dir /path/to/output
```

## 🗂️ Project Structure

```
OcrtoQA_for_EduLLM/
├── data/
│   ├── raw_pdfs/           # Input PDF files
│   └── processed/
│       ├── text/           # OCR text dumps (optional)
│       ├── latex/          # OCR LaTeX dumps (optional)
│       └── qa/             # Final JSONL output
├── src/
│   ├── ingestion.py        # PDF scanning and validation
│   ├── ocr.py              # OCR processing (Tesseract + MathPix)
│   ├── postprocess.py      # OCR cleanup and normalization
│   ├── content_filter.py   # Smart filtering of non-math content
│   ├── qa_chain.py         # QA generation with GPT-4o
│   ├── validator.py        # Mathematical and quality validation
│   └── pipeline.py         # Main orchestrator
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🛠️ Technical Stack

- **OCR**: Tesseract 5 (eng+kor+equ), MathPix API
- **LLM**: OpenAI GPT-4o via LangChain
- **Content Filtering**: Smart math content detection and noise removal
- **Validation**: SymPy (mathematical), BERTScore (quality)
- **ML**: Transformers, PyTorch
- **Processing**: OpenCV, pdf2image

## 📊 Output Format

Each JSONL file contains one QA pair per line:

```json
{
  "question": "Solve for x: 2x + 5 = 11",
  "answer": "3",
  "latex": "2x + 5 = 11",
  "valid_sympy": true,
  "bertscore_f1": 0.92
}
```

## 🐳 Docker Support

Build and run with Docker:

```bash
# Build image
docker build -t ocr-qa-pipeline .

# Run pipeline
docker run -v $(pwd)/data:/app/data \
           -e OPENAI_API_KEY=$OPENAI_API_KEY \
           -e MATHPIX_APP_ID=$MATHPIX_APP_ID \
           -e MATHPIX_APP_KEY=$MATHPIX_APP_KEY \
           ocr-qa-pipeline python src/pipeline.py
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for QA generation
- `MATHPIX_APP_ID`: Optional, for LaTeX extraction
- `MATHPIX_APP_KEY`: Optional, for LaTeX extraction

### Pipeline Parameters

Edit in `pipeline.py` or pass via CLI:
- `model_name`: LLM model (default: "gpt-4o")
- `bert_threshold`: Quality threshold (default: 0.80)
- `chunk_size`: Text chunk size for processing (default: 2000)

## 📈 Performance Tips

1. **PDF Quality**: Higher resolution PDFs yield better OCR results
2. **GPU Support**: Enable CUDA for faster BERTScore validation
3. **Batch Processing**: Process multiple PDFs in parallel using `--pdfs`
4. **API Limits**: Be mindful of OpenAI/MathPix rate limits

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License.