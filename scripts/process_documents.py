"""
Script to process all PDF documents and build the vector store.
"""
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_processing.document_processor import DocumentProcessor
from src.document_processing.chunker import SemanticChunker
from src.retrieval.vector_store import VectorStore
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_all_documents(
    data_dir: Path,
    output_dir: Path,
    extract_tables_charts: bool = True,
    clear_existing: bool = False
):
    """
    Process all PDF documents in the data directory.

    Args:
        data_dir: Directory containing PDF files
        output_dir: Directory to save processed documents
        extract_tables_charts: Whether to extract tables/charts with VLM
        clear_existing: Whether to clear existing vector store
    """
    logger.info("=" * 80)
    logger.info("Starting document processing pipeline")
    logger.info("=" * 80)

    # Find all PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        logger.error("No PDF files found in data directory")
        return

    # Initialize components
    logger.info("Initializing components...")
    document_processor = DocumentProcessor(use_vlm=extract_tables_charts)
    chunker = SemanticChunker()
    vector_store = VectorStore()

    # Clear existing data if requested
    if clear_existing:
        logger.info("Clearing existing vector store...")
        vector_store.clear()

    # Process each document
    all_chunks = []

    for pdf_file in tqdm(pdf_files, desc="Processing documents"):
        try:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing: {pdf_file.name}")
            logger.info(f"{'=' * 80}")

            # Check if already processed
            processed_file = output_dir / f"{pdf_file.name}.json"
            if processed_file.exists() and not clear_existing:
                logger.info(f"Loading existing processed document: {processed_file}")
                document = document_processor.load_processed_document(processed_file)
            else:
                # Process document
                logger.info("Running document processing pipeline...")
                document = document_processor.process_document(
                    pdf_file,
                    extract_tables_charts=extract_tables_charts
                )

                # Save processed document
                document_processor.save_processed_document(document, output_dir)

            # Create chunks
            logger.info("Creating semantic chunks...")
            chunks = chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks")

            # Save chunks
            chunks_file = output_dir / f"{pdf_file.name}.chunks.json"
            chunker.save_chunks(chunks, chunks_file)

            # Convert to dictionaries for vector store
            chunk_dicts = [
                {
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text,
                    'paper_name': chunk.paper_name,
                    'page_num': chunk.page_num,
                    'region_id': chunk.region_id,
                    'region_type': chunk.region_type,
                    'bbox': chunk.bbox,
                    'reading_order': chunk.reading_order,
                    'chunk_index': chunk.chunk_index,
                    'section': chunk.section
                }
                for chunk in chunks
            ]

            all_chunks.extend(chunk_dicts)

            logger.info(f"✓ Successfully processed {pdf_file.name}")

            # Memory cleanup after each document
            del document
            del chunks
            del chunk_dicts
            import gc
            gc.collect()
            logger.info("Memory cleaned up")

        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_file.name}: {e}", exc_info=True)
            # Clean up on error too
            import gc
            gc.collect()
            continue

    # Add all chunks to vector store
    if all_chunks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
        logger.info(f"{'=' * 80}")

        try:
            vector_store.add_chunks(all_chunks)
            logger.info("✓ Successfully added all chunks to vector store")

            # Display statistics
            stats = vector_store.get_stats()
            logger.info(f"\n{'=' * 80}")
            logger.info("Vector Store Statistics")
            logger.info(f"{'=' * 80}")
            logger.info(f"Total chunks: {stats['total_chunks']}")
            logger.info(f"Total papers: {len(stats['papers'])}")
            logger.info("\nPapers:")
            for paper in stats['papers']:
                logger.info(f"  • {paper['name']}: {paper['chunks']} chunks")
            logger.info("\nRegion types:")
            for region_type, count in stats['region_type_counts'].items():
                logger.info(f"  • {region_type}: {count} chunks")

        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}", exc_info=True)

    logger.info(f"\n{'=' * 80}")
    logger.info("Document processing pipeline completed!")
    logger.info(f"{'=' * 80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process research papers and build vector store"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=config.DATA_DIR,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Directory to save processed documents"
    )
    parser.add_argument(
        '--no-vlm',
        action='store_true',
        help="Skip VLM extraction for tables/charts"
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help="Clear existing vector store before processing"
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process documents
    process_all_documents(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        extract_tables_charts=not args.no_vlm,
        clear_existing=args.clear
    )


if __name__ == '__main__':
    main()
