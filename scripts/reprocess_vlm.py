"""
Reprocess VLM extractions for existing documents.
This script re-extracts tables and figures using the improved VLM prompts.
"""
import logging
from pathlib import Path
from tqdm import tqdm
import json

from src.document_processing.document_processor import DocumentProcessor, NumpyEncoder
from src.document_processing.vlm_extractor import VLMExtractor
from src.config import config
from pdf2image import convert_from_path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reprocess_vlm_extractions(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    pdf_dir: Path = config.DATA_DIR,
    limit: int = None
):
    """
    Reprocess VLM extractions for existing documents.

    Args:
        processed_dir: Directory with processed .json files
        pdf_dir: Directory with original PDF files
        limit: Optional limit on number of documents to process
    """
    logger.info("=" * 80)
    logger.info("Reprocessing VLM extractions from existing documents")
    logger.info("=" * 80)

    # Find all processed JSON files (not chunks files)
    json_files = [f for f in processed_dir.glob("*.json") if not f.name.endswith(".chunks.json")]

    if limit:
        json_files = json_files[:limit]

    logger.info(f"Found {len(json_files)} processed documents")

    if not json_files:
        logger.error("No processed documents found!")
        return

    # Initialize VLM extractor
    vlm_extractor = VLMExtractor()
    document_processor = DocumentProcessor(use_vlm=False, use_tesseract=True)

    # Process each document
    successful = 0
    failed = 0
    total_extractions = 0
    successful_extractions = 0

    for json_file in tqdm(json_files, desc="Reprocessing VLM extractions"):
        try:
            logger.info(f"\nProcessing: {json_file.name}")

            # Load processed document
            with open(json_file, 'r', encoding='utf-8') as f:
                document = json.load(f)

            # Find corresponding PDF
            pdf_name = document['filename']
            pdf_path = pdf_dir / pdf_name

            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                failed += 1
                continue

            # Convert PDF to images using document processor
            logger.info("Converting PDF to images...")
            try:
                images_list = convert_from_path(str(pdf_path), dpi=200)
                images = {i + 1: img for i, img in enumerate(images_list)}
            except FileNotFoundError as e:
                logger.error(f"pdftoppm not found. Install poppler: brew install poppler")
                logger.error(f"Skipping VLM reprocessing for {pdf_name}")
                failed += 1
                continue

            # Find table and figure regions
            table_figure_regions = []
            for page in document['pages']:
                for region in page['regions']:
                    if region['region_type'] in ['table', 'figure']:
                        table_figure_regions.append(region)

            logger.info(f"Found {len(table_figure_regions)} table/figure regions")

            if not table_figure_regions:
                logger.info("No tables/figures to process")
                successful += 1
                continue

            # Extract with VLM
            logger.info("Running VLM extraction...")
            vlm_extractions = vlm_extractor.process_regions(table_figure_regions, images)

            # Count successful extractions
            for region_id, extraction in vlm_extractions.items():
                total_extractions += 1
                if 'error' not in extraction or 'extraction failed' not in extraction.get('summary', '').lower():
                    # Check if it has actual data
                    if extraction.get('type') == 'table':
                        if extraction.get('headers') or extraction.get('rows'):
                            successful_extractions += 1
                    elif extraction.get('type') == 'chart':
                        if extraction.get('chart_type') != 'unknown':
                            successful_extractions += 1

            # Update document with new VLM extractions
            document['vlm_extractions'] = vlm_extractions

            # Save updated document
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(document, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            logger.info(f"✓ Successfully reprocessed {json_file.name}")
            successful += 1

            # Clean up images
            del images
            del images_list
            import gc
            gc.collect()

        except Exception as e:
            failed += 1
            logger.error(f"✗ Failed to process {json_file.name}: {e}", exc_info=True)
            continue

    logger.info(f"\n{'=' * 80}")
    logger.info("Reprocessing complete!")
    logger.info(f"{'=' * 80}")
    logger.info(f"Documents processed: {successful}/{len(json_files)}")
    logger.info(f"Documents failed: {failed}/{len(json_files)}")
    logger.info(f"Total VLM extractions: {total_extractions}")
    logger.info(f"Successful extractions: {successful_extractions}")
    if total_extractions > 0:
        logger.info(f"Success rate: {(successful_extractions/total_extractions*100):.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess VLM extractions")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    args = parser.parse_args()

    reprocess_vlm_extractions(limit=args.limit)
