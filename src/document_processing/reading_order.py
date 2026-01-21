"""
Reading order detection for multi-column and complex document layouts.
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderedRegion:
    """Container for region with reading order."""
    region_id: str
    region_type: str
    bbox: List[float]
    confidence: float
    page_num: int
    reading_order: int


class ReadingOrderDetector:
    """
    Custom heuristic-based reading order detection.

    Uses a combination of:
    - Column detection
    - Top-to-bottom, left-to-right ordering
    - Region type priority (title > text > table > figure)
    """

    def __init__(self):
        """Initialize reading order detector."""
        self.type_priority = {
            'title': 0,
            'text': 1,
            'list': 2,
            'table': 3,
            'figure': 4
        }

    def determine_reading_order(
        self,
        regions: List[Dict[str, Any]]
    ) -> List[OrderedRegion]:
        """
        Determine reading order for document regions.

        Args:
            regions: List of layout regions from layout detector

        Returns:
            List of regions with reading order assigned
        """
        if not regions:
            return []

        # Sort by page number first
        regions_by_page = {}
        for region in regions:
            page_num = region['page_num']
            if page_num not in regions_by_page:
                regions_by_page[page_num] = []
            regions_by_page[page_num].append(region)

        # Process each page
        ordered_regions = []
        global_order = 0

        for page_num in sorted(regions_by_page.keys()):
            page_regions = regions_by_page[page_num]
            page_ordered = self._order_page_regions(page_regions)

            # Assign global reading order
            for region_data in page_ordered:
                ordered_region = OrderedRegion(
                    region_id=region_data['region_id'],
                    region_type=region_data['region_type'],
                    bbox=region_data['bbox'],
                    confidence=region_data['confidence'],
                    page_num=region_data['page_num'],
                    reading_order=global_order
                )
                ordered_regions.append(ordered_region)
                global_order += 1

        logger.info(f"Assigned reading order to {len(ordered_regions)} regions")
        return ordered_regions

    def _order_page_regions(
        self,
        regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Order regions within a single page.

        Strategy:
        1. Detect columns based on horizontal position
        2. Within each column, sort by vertical position
        3. Apply type priority for regions at similar positions

        Args:
            regions: Regions from a single page

        Returns:
            Ordered list of regions
        """
        if not regions:
            return []

        # Detect columns
        columns = self._detect_columns(regions)

        # Sort columns left to right
        sorted_columns = sorted(columns, key=lambda col: col['x_center'])

        # Order regions within each column
        ordered_regions = []
        for column in sorted_columns:
            column_regions = column['regions']
            # Sort by: 1) type priority, 2) vertical position (top to bottom)
            column_regions.sort(key=lambda r: (
                self.type_priority.get(r['region_type'], 5),
                r['bbox'][1]  # y1 coordinate
            ))
            ordered_regions.extend(column_regions)

        return ordered_regions

    def _detect_columns(
        self,
        regions: List[Dict[str, Any]],
        column_threshold: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Detect columns in a page layout using horizontal clustering.

        Args:
            regions: List of regions
            column_threshold: Minimum horizontal distance to define separate columns

        Returns:
            List of column dictionaries with regions grouped
        """
        if not regions:
            return []

        # Calculate center x-coordinate for each region
        region_centers = []
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            x_center = (x1 + x2) / 2
            region_centers.append((x_center, region))

        # Sort by x-center
        region_centers.sort(key=lambda x: x[0])

        # Cluster into columns
        columns = []
        current_column = {
            'x_center': region_centers[0][0],
            'regions': [region_centers[0][1]]
        }

        for x_center, region in region_centers[1:]:
            # Check if this region belongs to current column
            if abs(x_center - current_column['x_center']) < column_threshold:
                current_column['regions'].append(region)
                # Update column center (weighted average)
                n = len(current_column['regions'])
                current_column['x_center'] = (
                    (current_column['x_center'] * (n - 1) + x_center) / n
                )
            else:
                # Start new column
                columns.append(current_column)
                current_column = {
                    'x_center': x_center,
                    'regions': [region]
                }

        # Add last column
        columns.append(current_column)

        logger.debug(f"Detected {len(columns)} columns")
        return columns

    @staticmethod
    def get_ordered_text(
        ordered_regions: List[OrderedRegion],
        region_text_map: Dict[str, str]
    ) -> str:
        """
        Get text from ordered regions.

        Args:
            ordered_regions: Regions with reading order
            region_text_map: Mapping from region_id to extracted text

        Returns:
            Concatenated text in reading order
        """
        texts = []
        for region in sorted(ordered_regions, key=lambda r: r.reading_order):
            text = region_text_map.get(region.region_id, '')
            if text:
                texts.append(text)
        return '\n\n'.join(texts)
