"""
Layout detection using LayoutParser to identify document regions.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import layoutparser as lp
from dataclasses import dataclass, asdict

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class LayoutRegion:
    """Container for layout region information."""
    region_id: str
    region_type: str  # text, title, list, table, figure
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    page_num: int


class LayoutDetector:
    """LayoutParser-based document layout detector."""

    def __init__(self):
        """Initialize layout detection model."""
        try:
            self.model = lp.PaddleDetectionLayoutModel(
                config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
                label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
            )
            self.confidence_threshold = config.LAYOUT_CONFIDENCE_THRESHOLD
            logger.info("Layout detection model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize layout model: {e}")
            raise

    def detect_layout(self, image: Image.Image, page_num: int) -> List[LayoutRegion]:
        """
        Detect layout regions in an image.

        Args:
            image: PIL Image object
            page_num: Page number

        Returns:
            List of detected layout regions
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Detect layout
        layout = self.model.detect(img_array)

        # Convert to LayoutRegion objects
        regions = []
        for idx, block in enumerate(layout):
            # Filter by confidence threshold
            if block.score >= self.confidence_threshold:
                region = LayoutRegion(
                    region_id=f"page{page_num}_region{idx}",
                    region_type=block.type,
                    bbox=[block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2],
                    confidence=block.score,
                    page_num=page_num
                )
                regions.append(region)

        logger.debug(f"Detected {len(regions)} regions on page {page_num}")
        return regions

    def process_pdf_pages(
        self,
        images: List[Image.Image],
        start_page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF page images for layout detection.

        Args:
            images: List of PIL Image objects
            start_page: Starting page number

        Returns:
            List of page results with layout regions
        """
        page_results = []

        for idx, image in enumerate(images):
            page_num = start_page + idx
            regions = self.detect_layout(image, page_num)

            page_results.append({
                'page_num': page_num,
                'width': image.width,
                'height': image.height,
                'regions': [asdict(r) for r in regions]
            })

        return page_results

    @staticmethod
    def filter_regions_by_type(
        regions: List[LayoutRegion],
        region_types: List[str]
    ) -> List[LayoutRegion]:
        """
        Filter regions by type.

        Args:
            regions: List of layout regions
            region_types: Types to filter for

        Returns:
            Filtered list of regions
        """
        return [r for r in regions if r.region_type in region_types]

    @staticmethod
    def get_region_area(bbox: List[float]) -> float:
        """Calculate area of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = LayoutDetector.get_region_area(bbox1)
        area2 = LayoutDetector.get_region_area(bbox2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
