"""
OCR engine using PaddleOCR for text extraction with bounding boxes and confidence scores.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from dataclasses import dataclass

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR results."""
    text: str
    bbox: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    page_num: int


class OCREngine:
    """PaddleOCR-based text extraction engine."""

    def __init__(self):
        """Initialize PaddleOCR engine."""
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=config.OCR_LANG
        )
        logger.info("OCR engine initialized")

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract text with bounding boxes.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page results with OCR data
        """
        logger.info(f"Processing PDF: {pdf_path.name}")

        # Convert PDF to images
        try:
            images = convert_from_path(str(pdf_path), dpi=300)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

        page_results = []
        for page_num, image in enumerate(images, start=1):
            logger.debug(f"Processing page {page_num}/{len(images)}")
            page_result = self._process_page(image, page_num)
            page_results.append(page_result)

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
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run OCR
        ocr_results = self.ocr.ocr(img_array)

        # Parse results
        ocr_data = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]

                if confidence >= config.OCR_CONFIDENCE_THRESHOLD:
                    ocr_data.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence,
                        'page_num': page_num
                    })

        return {
            'page_num': page_num,
            'width': image.width,
            'height': image.height,
            'ocr_results': ocr_data
        }

    def get_text_from_bbox(
        self,
        page_result: Dict[str, Any],
        target_bbox: List[float]
    ) -> str:
        """
        Extract text from a specific bounding box region.

        Args:
            page_result: Page OCR result
            target_bbox: Target bounding box [x1, y1, x2, y2]

        Returns:
            Concatenated text from the region
        """
        texts = []
        x1, y1, x2, y2 = target_bbox

        for ocr_item in page_result['ocr_results']:
            # Get center point of OCR bbox
            bbox = ocr_item['bbox']
            cx = sum(p[0] for p in bbox) / 4
            cy = sum(p[1] for p in bbox) / 4

            # Check if center is within target bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                texts.append(ocr_item['text'])

        return ' '.join(texts)
