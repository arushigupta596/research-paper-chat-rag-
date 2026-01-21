"""
Simplified document processing script - processes PDFs without complex dependencies.
"""
import logging
import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_pdf_to_text(pdf_path: Path) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader

    logger.info(f"Processing: {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
        text_content = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_content.append(f"\n\n--- Page {page_num} ---\n\n{text}")

        full_text = '\n'.join(text_content)
        logger.info(f"✓ Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        return full_text

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return ""


def create_chunks(text: str, paper_name: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Create simple text chunks."""
    chunks = []
    words = text.split()

    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end])

        chunks.append({
            'chunk_id': f"{paper_name}_chunk{chunk_id}",
            'text': chunk_text,
            'paper_name': paper_name,
            'page_num': 1,  # Simplified
            'region_id': f"{paper_name}_region{chunk_id}",
            'region_type': 'text',
            'bbox': [0, 0, 100, 100],  # Dummy bbox
            'reading_order': chunk_id,
            'chunk_index': chunk_id,
            'section': None
        })

        start = end - overlap
        chunk_id += 1

        if end >= len(words):
            break

    return chunks


def main():
    """Main processing function."""
    print("=" * 80)
    print("Simplified Document Processing")
    print("=" * 80)
    print()

    # Find PDF files
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    if not pdf_files:
        print("No PDF files found!")
        return

    # Initialize ChromaDB
    from src.retrieval.vector_store import VectorStore

    print("Initializing vector store...")
    vector_store = VectorStore()

    # Clear existing
    print("Clearing existing data...")
    vector_store.clear()

    all_chunks = []

    # Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        # Extract text
        text = simple_pdf_to_text(pdf_file)

        if not text:
            continue

        # Create chunks
        paper_name = pdf_file.name
        chunks = create_chunks(text, paper_name)

        print(f"  Created {len(chunks)} chunks from {paper_name}")
        all_chunks.extend(chunks)

    # Add to vector store
    if all_chunks:
        print(f"\nAdding {len(all_chunks)} chunks to vector store...")
        vector_store.add_chunks(all_chunks)

        # Show stats
        stats = vector_store.get_stats()
        print("\n" + "=" * 80)
        print("Processing Complete!")
        print("=" * 80)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total papers: {len(stats['papers'])}")
        print("\nPapers:")
        for paper in stats['papers']:
            print(f"  • {paper['name']}: {paper['chunks']} chunks")

    print("\n✓ Ready to use! Run: streamlit run app.py")


if __name__ == '__main__':
    main()
