#!/usr/bin/env python3
"""
Installation test script for OCR to QA Pipeline.

Run this script to verify all dependencies are correctly installed.
"""

import sys
import subprocess


def test_import(module_name, package_name=None):
    """Test if a Python module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def test_command(command, name):
    """Test if a system command is available."""
    try:
        subprocess.run([command, "--version"], 
                      capture_output=True, 
                      check=True)
        print(f"✓ {name} is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✗ {name} is NOT installed")
        return False


def test_tesseract_languages():
    """Test if required Tesseract languages are installed."""
    try:
        result = subprocess.run(["tesseract", "--list-langs"], 
                              capture_output=True, 
                              text=True)
        langs = result.stdout.lower()
        
        required_langs = ["eng", "kor"]
        for lang in required_langs:
            if lang in langs:
                print(f"  ✓ Tesseract language '{lang}' is installed")
            else:
                print(f"  ✗ Tesseract language '{lang}' is NOT installed")
                
    except Exception as e:
        print(f"  ✗ Could not check Tesseract languages: {e}")


def main():
    print("=" * 60)
    print("OCR to QA Pipeline - Installation Test")
    print("=" * 60)
    
    all_ok = True
    
    # Test system dependencies
    print("\n1. Testing system dependencies:")
    all_ok &= test_command("tesseract", "Tesseract OCR")
    if all_ok:
        test_tesseract_languages()
    
    # Test Python packages
    print("\n2. Testing Python packages:")
    packages = [
        ("pytesseract", None),
        ("pdf2image", None),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("sympy", None),
        ("numpy", None),
        ("langchain", None),
        ("openai", None),
        ("bert_score", "bert-score"),
        ("transformers", None),
        ("torch", None),
    ]
    
    for module, package in packages:
        all_ok &= test_import(module, package)
    
    # Test environment variables
    print("\n3. Testing environment variables:")
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY is set")
    else:
        print("✗ OPENAI_API_KEY is NOT set (required for QA generation)")
        all_ok = False
    
    if os.getenv("MATHPIX_APP_ID") and os.getenv("MATHPIX_APP_KEY"):
        print("✓ MathPix API credentials are set")
    else:
        print("⚠ MathPix API credentials are NOT set (optional)")
    
    # Test local modules
    print("\n4. Testing local modules:")
    sys.path.insert(0, "src")
    
    local_modules = [
        "ingestion",
        "ocr",
        "postprocess",
        "qa_chain",
        "validator",
        "pipeline"
    ]
    
    for module in local_modules:
        all_ok &= test_import(module)
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All required components are installed!")
        print("\nYou can now run the pipeline with:")
        print("  python src/pipeline.py")
    else:
        print("✗ Some components are missing.")
        print("\nPlease:")
        print("1. Install missing system dependencies")
        print("2. Run: pip install -r requirements.txt")
        print("3. Set up your .env file with API keys")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())