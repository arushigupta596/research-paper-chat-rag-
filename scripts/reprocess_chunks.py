"""
Reprocess already-processed documents: create chunks and add to ChromaDB.
This script loads existing .json processed documents and completes the chunking/embedding pipeline.
"""
import logging
from pathlib import Path
from tqdm import tqdm

from src.document_processing.document_processor import DocumentProcessor
from src.document_processing.chunker import SemanticChunker
from src.retrieval.vector_store import VectorStore
from src.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reprocess_chunks(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    clear_existing: bool = False
):
    """
    Reprocess existing documents to create chunks and embeddings.

    Args:
        processed_dir: Directory with processed .json files
        clear_existing: Whether to clear existing ChromaDB
    """
    logger.info("=" * 80)
    logger.info("Reprocessing chunks and embeddings from existing documents")
    logger.info("=" * 80)

    # Find all processed JSON files (not chunks files)
    json_files = [f for f in processed_dir.glob("*.json") if not f.name.endswith(".chunks.json")]
    logger.info(f"Found {len(json_files)} processed documents")

    if not json_files:
        logger.error("No processed documents found!")
        return

    # Initialize components
    document_processor = DocumentProcessor()
    chunker = SemanticChunker()
    vector_store = VectorStore()

    # Clear if requested
    if clear_existing:
        logger.info("Clearing existing ChromaDB...")
        vector_store.clear()

    # Process each document
    all_chunks = []
    successful = 0
    failed = 0

    for json_file in tqdm(json_files, desc="Reprocessing documents"):
        try:
            logger.info(f"\nProcessing: {json_file.name}")

            # Load processed document
            document = document_processor.load_processed_document(json_file)

            # Create chunks
            logger.info("Creating semantic chunks...")
            chunks = chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks")

            # Save chunks
            chunks_file = processed_dir / f"{json_file.name}.chunks.json"
            chunker.save_chunks(chunks, chunks_file)
            logger.info(f"Saved chunks to {chunks_file}")

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
            successful += 1
            logger.info(f"✓ Successfully processed {json_file.name}")

        except Exception as e:
            failed += 1
            logger.error(f"✗ Failed to process {json_file.name}: {e}", exc_info=True)
            continue

    # Add all chunks to vector store
    if all_chunks:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Adding {len(all_chunks)} chunks to ChromaDB...")
        logger.info(f"{'=' * 80}")

        try:
            vector_store.add_chunks(all_chunks)
            logger.info("✓ Successfully added all chunks to ChromaDB")

            # Display statistics
            stats = vector_store.get_stats()
            logger.info(f"\n{'=' * 80}")
            logger.info("ChromaDB Statistics")
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
            logger.error(f"Failed to add chunks to ChromaDB: {e}", exc_info=True)
    else:
        logger.warning("No chunks to add to ChromaDB!")

    logger.info(f"\n{'=' * 80}")
    logger.info("Reprocessing complete!")
    logger.info(f"{'=' * 80}")
    logger.info(f"Successful: {successful}/{len(json_files)}")
    logger.info(f"Failed: {failed}/{len(json_files)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess chunks and embeddings")
    parser.add_argument("--clear", action="store_true", help="Clear existing ChromaDB")
    args = parser.parse_args()

    reprocess_chunks(clear_existing=args.clear)
