"""
Test script to verify installation and system setup.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    packages = {
        'streamlit': 'Streamlit',
        'paddleocr': 'PaddleOCR',
        'layoutparser': 'LayoutParser',
        'chromadb': 'ChromaDB',
        'langchain': 'LangChain',
        'sentence_transformers': 'SentenceTransformers',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'pdf2image': 'pdf2image',
        'pypdf': 'PyPDF',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
    }

    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully")
        return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    try:
        from src.config import config

        print(f"✓ Config loaded")
        print(f"  - Data directory: {config.DATA_DIR}")
        print(f"  - Processed data directory: {config.PROCESSED_DATA_DIR}")
        print(f"  - Chroma directory: {config.CHROMA_PERSIST_DIR}")

        # Check API key
        if config.OPENROUTER_API_KEY:
            print(f"✓ OpenRouter API key configured")
        else:
            print("⚠️  OpenRouter API key not configured (required for VLM and LLM)")

        return True

    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")

    from src.config import config

    directories = [
        ('Data directory', config.DATA_DIR),
        ('Processed data directory', config.PROCESSED_DATA_DIR),
        ('Chroma directory', config.CHROMA_PERSIST_DIR),
        ('Logs directory', config.LOGS_DIR),
    ]

    all_exist = True

    for name, path in directories:
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name} not found: {path}")
            all_exist = False

    return all_exist


def test_pdf_files():
    """Check for PDF files in data directory."""
    print("\nChecking for PDF files...")

    from src.config import config

    pdf_files = list(config.DATA_DIR.glob("*.pdf"))

    if pdf_files:
        print(f"✓ Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files[:5]:  # Show first 5
            print(f"  - {pdf_file.name}")
        if len(pdf_files) > 5:
            print(f"  ... and {len(pdf_files) - 5} more")
        return True
    else:
        print("⚠️  No PDF files found in Data directory")
        print("   Place your research papers in the Data/ directory")
        return False


def test_components():
    """Test that core components can be initialized."""
    print("\nTesting core components...")

    try:
        # Test OCR
        print("Testing OCR engine...")
        from src.document_processing.ocr_engine import OCREngine
        ocr = OCREngine()
        print("✓ OCR engine initialized")

        # Test Layout Detector
        print("Testing layout detector...")
        from src.document_processing.layout_detector import LayoutDetector
        layout = LayoutDetector()
        print("✓ Layout detector initialized")

        # Test Vector Store
        print("Testing vector store...")
        from src.retrieval.vector_store import VectorStore
        vector_store = VectorStore()
        print("✓ Vector store initialized")
        print(f"  - Current collection size: {vector_store.collection.count()} chunks")

        return True

    except Exception as e:
        print(f"✗ Component initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("System Installation Test")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Directories", test_directories()))
    results.append(("PDF Files", test_pdf_files()))
    results.append(("Core Components", test_components()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Ensure OpenRouter API key is configured in .env")
        print("2. Run: python scripts/process_documents.py")
        print("3. Run: streamlit run app.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("- Missing packages: Run 'pip install -r requirements.txt'")
        print("- Missing API key: Add OPENROUTER_API_KEY to .env file")
        print("- Missing PDF files: Place papers in Data/ directory")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
