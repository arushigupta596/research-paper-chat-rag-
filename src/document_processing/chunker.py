"""
Semantic chunking with metadata enrichment for knowledge representation.
"""
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

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


@dataclass
class Chunk:
    """Container for a semantic chunk with metadata."""
    chunk_id: str
    text: str
    paper_name: str
    page_num: int
    region_id: str
    region_type: str
    bbox: List[float]
    reading_order: int
    section: Optional[str] = None
    chunk_index: int = 0
    metadata: Optional[Dict[str, Any]] = None


class SemanticChunker:
    """
    Semantic chunking that respects document structure.

    Chunks based on:
    - Reading order
    - Region boundaries
    - Section boundaries
    - Token limits
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum chunk size in tokens (approx chars/4)
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        logger.info(f"Chunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_document(
        self,
        document: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk a processed document into semantic chunks.

        Args:
            document: Processed document structure

        Returns:
            List of chunks with metadata
        """
        logger.info(f"Chunking document: {document['filename']}")

        chunks = []
        ordered_regions = document['ordered_regions']
        vlm_extractions = document.get('vlm_extractions', {})
        paper_name = document['filename']
        pages = document['pages']

        # Create region lookup map: {region_id: region_data_with_text}
        region_lookup = {}
        for page in pages:
            for region in page['regions']:
                region_lookup[region['region_id']] = region

        # Process each region in reading order
        for region in sorted(ordered_regions, key=lambda r: r['reading_order']):
            region_id = region['region_id']
            region_type = region['region_type']

            # Get text content
            if region_type in ['table', 'figure'] and region_id in vlm_extractions:
                # Use VLM-extracted structured data
                text = self._format_vlm_extraction(vlm_extractions[region_id])
            else:
                # Look up text from pages structure
                region_with_text = region_lookup.get(region_id)
                if region_with_text:
                    text = region_with_text.get('text', '')
                else:
                    text = ''

            if not text or len(text.strip()) < 10:
                continue

            # Create chunks from this region
            region_chunks = self._chunk_text(
                text=text,
                paper_name=paper_name,
                page_num=region['page_num'],
                region_id=region_id,
                region_type=region_type,
                bbox=region['bbox'],
                reading_order=region['reading_order']
            )

            chunks.extend(region_chunks)

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _chunk_text(
        self,
        text: str,
        paper_name: str,
        page_num: int,
        region_id: str,
        region_type: str,
        bbox: List[float],
        reading_order: int
    ) -> List[Chunk]:
        """
        Split text into chunks respecting size limits.

        Args:
            text: Text to chunk
            paper_name: Name of the paper
            page_num: Page number
            region_id: Region identifier
            region_type: Type of region
            bbox: Bounding box
            reading_order: Reading order

        Returns:
            List of chunks
        """
        chunks = []

        # Approximate tokens (chars / 4)
        approx_tokens = len(text) / 4

        if approx_tokens <= self.chunk_size:
            # Single chunk
            chunk = Chunk(
                chunk_id=f"{paper_name}_{region_id}_chunk0",
                text=text,
                paper_name=paper_name,
                page_num=page_num,
                region_id=region_id,
                region_type=region_type,
                bbox=bbox,
                reading_order=reading_order,
                chunk_index=0
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks with overlap
            char_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4

            start = 0
            chunk_index = 0

            while start < len(text):
                end = start + char_size

                # Find natural break point (sentence/paragraph)
                if end < len(text):
                    # Look for period, newline, or space
                    for break_char in ['\n\n', '\n', '. ', ' ']:
                        break_pos = text.rfind(break_char, start, end)
                        if break_pos != -1:
                            end = break_pos + len(break_char)
                            break

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk = Chunk(
                        chunk_id=f"{paper_name}_{region_id}_chunk{chunk_index}",
                        text=chunk_text,
                        paper_name=paper_name,
                        page_num=page_num,
                        region_id=region_id,
                        region_type=region_type,
                        bbox=bbox,
                        reading_order=reading_order,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Move start forward with overlap
                start = end - char_overlap if end < len(text) else end

                # Prevent infinite loop
                if start >= len(text):
                    break

        return chunks

    def _format_vlm_extraction(self, vlm_data: Dict[str, Any]) -> str:
        """
        Format VLM-extracted structured data into readable text.

        Args:
            vlm_data: VLM extraction result

        Returns:
            Formatted text representation
        """
        data_type = vlm_data.get('type', 'unknown')

        if data_type == 'table':
            return self._format_table(vlm_data)
        elif data_type == 'chart':
            return self._format_chart(vlm_data)
        else:
            return json.dumps(vlm_data, indent=2)

    def _format_table(self, table_data: Dict[str, Any]) -> str:
        """Format table data into text."""
        parts = [f"[TABLE]"]

        if table_data.get('summary'):
            parts.append(f"Summary: {table_data['summary']}")

        # Format headers and rows
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])

        if headers:
            parts.append(f"Headers: {', '.join(headers)}")

        if rows:
            parts.append("Data:")
            for i, row in enumerate(rows[:10]):  # Limit to first 10 rows
                parts.append(f"  Row {i + 1}: {', '.join(map(str, row))}")

        if table_data.get('units'):
            parts.append(f"Units: {table_data['units']}")

        if table_data.get('footnotes'):
            parts.append(f"Notes: {table_data['footnotes']}")

        return '\n'.join(parts)

    def _format_chart(self, chart_data: Dict[str, Any]) -> str:
        """Format chart data into text."""
        parts = [f"[CHART: {chart_data.get('chart_type', 'unknown')}]"]

        if chart_data.get('title'):
            parts.append(f"Title: {chart_data['title']}")

        if chart_data.get('summary'):
            parts.append(f"Summary: {chart_data['summary']}")

        # X and Y axes
        if chart_data.get('x_axis'):
            x_axis = chart_data['x_axis']
            parts.append(f"X-axis: {x_axis.get('label', 'N/A')}")

        if chart_data.get('y_axis'):
            y_axis = chart_data['y_axis']
            parts.append(f"Y-axis: {y_axis.get('label', 'N/A')} (Range: {y_axis.get('range', 'N/A')})")

        # Trends and insights
        if chart_data.get('trends'):
            parts.append(f"Trends: {chart_data['trends']}")

        if chart_data.get('key_insights'):
            insights = chart_data['key_insights']
            parts.append("Key Insights:")
            for insight in insights:
                parts.append(f"  - {insight}")

        if chart_data.get('anomalies'):
            parts.append(f"Anomalies: {chart_data['anomalies']}")

        return '\n'.join(parts)

    def save_chunks(self, chunks: List[Chunk], output_path: Path):
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunks
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = [asdict(chunk) for chunk in chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """
        Load chunks from JSON file.

        Args:
            input_path: Input file path

        Returns:
            List of chunks
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        chunks = [Chunk(**chunk_dict) for chunk_dict in chunks_data]

        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks
