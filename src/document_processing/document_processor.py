"""
Main document processor integrating OCR, layout detection, reading order, and VLM extraction.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from pdf2image import convert_from_path
from dataclasses import asdict

from .tesseract_ocr import TesseractOCR
import gc
from .layout_detector import LayoutDetector
from .reading_order import ReadingOrderDetector
from .vlm_extractor import VLMExtractor
from ..config import config

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DocumentProcessor:
    """
    Unified document processing pipeline.

    Orchestrates OCR, layout detection, reading order, and VLM extraction.
    """

    def __init__(self, use_vlm: bool = True, use_tesseract: bool = True):
        """Initialize all processing components."""
        self.ocr_engine = TesseractOCR() if use_tesseract else None
        self.layout_detector = LayoutDetector()
        self.reading_order_detector = ReadingOrderDetector()
        self.vlm_extractor = VLMExtractor() if use_vlm else None
        logger.info(f"Document processor initialized (VLM: {use_vlm}, Tesseract: {use_tesseract})")

    def process_document(
        self,
        pdf_path: Path,
        extract_tables_charts: bool = True
    ) -> Dict[str, Any]:
        """
        Process a complete PDF document.

        Args:
            pdf_path: Path to PDF file
            extract_tables_charts: Whether to use VLM for table/chart extraction

        Returns:
            Complete document structure with all extracted data
        """
        logger.info(f"Processing document: {pdf_path.name}")

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=300)
        logger.info(f"Converted {len(images)} pages to images")

        # Step 1: OCR extraction
        logger.info("Running OCR...")
        ocr_results = self.ocr_engine.process_pdf(pdf_path)

        # Step 2: Layout detection
        logger.info("Detecting layout...")
        layout_results = self.layout_detector.process_pdf_pages(images)

        # Step 3: Merge OCR with layout regions
        logger.info("Merging OCR with layout...")
        merged_pages = self._merge_ocr_and_layout(ocr_results, layout_results)

        # Step 4: Determine reading order
        logger.info("Determining reading order...")
        all_regions = []
        for page in merged_pages:
            all_regions.extend(page['regions'])

        ordered_regions = self.reading_order_detector.determine_reading_order(all_regions)

        # Step 5: VLM extraction for tables and charts
        vlm_data = {}
        if extract_tables_charts and self.vlm_extractor is not None:
            logger.info("Extracting tables and charts with VLM...")
            # Filter for table and figure regions
            table_chart_regions = [
                r for r in all_regions
                if r['region_type'] in ['table', 'figure']
            ]

            # Create image dictionary
            image_dict = {i + 1: img for i, img in enumerate(images)}

            vlm_data = self.vlm_extractor.process_regions(
                table_chart_regions,
                image_dict
            )

        # Step 6: Build final document structure
        document = {
            'filename': pdf_path.name,
            'num_pages': len(merged_pages),
            'pages': merged_pages,
            'ordered_regions': [asdict(r) for r in ordered_regions],
            'vlm_extractions': vlm_data,
            'metadata': {
                'total_regions': len(all_regions),
                'total_tables': len([r for r in all_regions if r['region_type'] == 'table']),
                'total_figures': len([r for r in all_regions if r['region_type'] == 'figure']),
                'vlm_extracted': len(vlm_data)
            }
        }

        logger.info(f"Document processing complete: {pdf_path.name}")

        # Clean up memory
        del images
        del merged_pages
        del all_regions
        gc.collect()

        return document

    def _merge_ocr_and_layout(
        self,
        ocr_results: List[Dict[str, Any]],
        layout_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge OCR text with layout regions.

        Args:
            ocr_results: OCR results by page
            layout_results: Layout detection results by page

        Returns:
            Merged page data with text in each region
        """
        merged_pages = []

        for ocr_page, layout_page in zip(ocr_results, layout_results):
            page_num = ocr_page['page_num']

            # Process each layout region
            enriched_regions = []
            for region in layout_page['regions']:
                # Extract text from OCR results within this region's bbox
                region_text = self._extract_text_from_region(
                    ocr_page['ocr_results'],
                    region['bbox']
                )

                # Enrich region with text
                enriched_region = {
                    **region,
                    'text': region_text
                }
                enriched_regions.append(enriched_region)

            merged_pages.append({
                'page_num': page_num,
                'width': layout_page['width'],
                'height': layout_page['height'],
                'regions': enriched_regions
            })

        return merged_pages

    def _extract_text_from_region(
        self,
        ocr_results: List[Dict[str, Any]],
        region_bbox: List[float]
    ) -> str:
        """
        Extract OCR text that falls within a region's bounding box.

        Args:
            ocr_results: List of OCR results
            region_bbox: Region bounding box [x1, y1, x2, y2]

        Returns:
            Concatenated text from the region
        """
        x1, y1, x2, y2 = region_bbox
        texts = []

        for ocr_item in ocr_results:
            # Get center point of OCR bbox
            bbox = ocr_item['bbox']
            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4

            # Check if center is within region bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                texts.append(ocr_item['text'])

        return ' '.join(texts)

    def save_processed_document(
        self,
        document: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Save processed document to JSON file.

        Args:
            document: Processed document structure
            output_dir: Output directory

        Returns:
            Path to saved JSON file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{document['filename']}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"Saved processed document to {output_path}")
        return output_path

    def load_processed_document(self, json_path: Path) -> Dict[str, Any]:
        """
        Load processed document from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Processed document structure
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        logger.info(f"Loaded processed document from {json_path}")
        return document
