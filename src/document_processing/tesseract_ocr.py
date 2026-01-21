"""
Lightweight OCR engine using Tesseract for text extraction.
Memory-efficient alternative to PaddleOCR for systems with limited RAM.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from dataclasses import dataclass
import gc

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR results."""
    text: str
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    page_num: int


class TesseractOCR:
    """Lightweight Tesseract-based text extraction engine."""

    def __init__(self):
        """Initialize Tesseract OCR engine."""
        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR engine initialized")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise

    def process_pdf(self, pdf_path: Path, batch_size: int = 3) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract text with bounding boxes.
        Processes in small batches to reduce memory usage.

        Args:
            pdf_path: Path to PDF file
            batch_size: Number of pages to process at once (smaller = less memory)

        Returns:
            List of page results with OCR data
        """
        logger.info(f"Processing PDF: {pdf_path.name}")

        # Get page count first
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            total_pages = info.get('Pages', 0)
            logger.info(f"Total pages: {total_pages}")
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            total_pages = None

        page_results = []

        # Process in batches to save memory
        if total_pages:
            for start_page in range(1, total_pages + 1, batch_size):
                end_page = min(start_page + batch_size - 1, total_pages)
                logger.info(f"Processing pages {start_page}-{end_page}/{total_pages}")

                try:
                    # Convert only this batch of pages
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=200,  # Lower DPI for less memory
                        first_page=start_page,
                        last_page=end_page
                    )

                    # Process each page in the batch
                    for idx, image in enumerate(images):
                        page_num = start_page + idx
                        logger.debug(f"OCR on page {page_num}")
                        page_result = self._process_page(image, page_num)
                        page_results.append(page_result)

                        # Clean up image immediately
                        del image

                    # Clean up batch
                    del images
                    gc.collect()

                except Exception as e:
                    logger.error(f"Failed to process pages {start_page}-{end_page}: {e}")
        else:
            # Fallback: convert all at once (less memory efficient)
            try:
                images = convert_from_path(str(pdf_path), dpi=200)
                for page_num, image in enumerate(images, start=1):
                    logger.debug(f"Processing page {page_num}/{len(images)}")
                    page_result = self._process_page(image, page_num)
                    page_results.append(page_result)
                    del image
                del images
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
                return []

        logger.info(f"Completed OCR for {len(page_results)} pages")
        return page_results

    def _process_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """
        Process a single page image with OCR.

        Args:
            image: PIL Image object
            page_num: Page number

        Returns:
            Dictionary containing page results
        """
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang='eng'
        )

        # Parse results
        ocr_results = []
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            # Filter out low confidence and empty text
            if conf > 0 and text:
                x, y, w, h = (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                )

                # Create bbox in the format expected [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]

                ocr_results.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf / 100.0  # Normalize to 0-1
                })

        return {
            'page_num': page_num,
            'ocr_results': ocr_results
        }

    def process_image(self, image: Image.Image, page_num: int = 1) -> Dict[str, Any]:
        """
        Process a single image with OCR.

        Args:
            image: PIL Image object
            page_num: Page number (default 1)

        Returns:
            Dictionary containing OCR results
        """
        return self._process_page(image, page_num)
