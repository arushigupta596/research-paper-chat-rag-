"""
Test the improved VLM extraction with better prompts and JSON cleaning.
"""
import logging
from PIL import Image
import json
from src.document_processing.vlm_extractor import VLMExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_vlm_extraction():
    """Test VLM extraction with improved prompts."""

    logger.info("=" * 80)
    logger.info("Testing Improved VLM Extraction")
    logger.info("=" * 80)

    # Initialize VLM extractor
    vlm_extractor = VLMExtractor()

    # Create a simple test image (white background)
    test_img = Image.new('RGB', (400, 300), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_img)

    # Draw a simple table
    draw.text((50, 50), "Name | Score | Grade", fill='black')
    draw.text((50, 80), "Alice | 95 | A", fill='black')
    draw.text((50, 110), "Bob | 87 | B", fill='black')
    draw.text((50, 140), "Charlie | 92 | A", fill='black')

    logger.info("\n--- Testing Table Extraction ---")

    try:
        # Test table extraction
        table_result = vlm_extractor.extract_table(
            image=test_img,
            region_id="test_table_1",
            page_num=1
        )

        logger.info(f"\n✓ Table Extraction Result:")
        logger.info(json.dumps(table_result, indent=2))

        if 'error' in table_result:
            logger.warning(f"⚠ Extraction had error: {table_result['error']}")
        else:
            logger.info("✓ No errors in extraction")

        if table_result.get('headers'):
            logger.info(f"✓ Successfully extracted headers: {table_result['headers']}")
        else:
            logger.warning("⚠ No headers extracted")

        if table_result.get('rows'):
            logger.info(f"✓ Successfully extracted {len(table_result['rows'])} rows")
        else:
            logger.warning("⚠ No rows extracted")

    except Exception as e:
        logger.error(f"✗ Table extraction failed: {e}", exc_info=True)

    # Test chart extraction
    logger.info("\n--- Testing Chart Extraction ---")

    # Create a simple chart image
    chart_img = Image.new('RGB', (400, 300), color='white')
    draw_chart = ImageDraw.Draw(chart_img)
    draw_chart.text((150, 20), "Sales by Month", fill='black')
    draw_chart.rectangle([50, 200, 80, 250], outline='blue', fill='lightblue')
    draw_chart.rectangle([100, 180, 130, 250], outline='blue', fill='lightblue')
    draw_chart.rectangle([150, 160, 180, 250], outline='blue', fill='lightblue')

    try:
        # Test chart extraction
        chart_result = vlm_extractor.extract_chart(
            image=chart_img,
            region_id="test_chart_1",
            page_num=1
        )

        logger.info(f"\n✓ Chart Extraction Result:")
        logger.info(json.dumps(chart_result, indent=2))

        if 'error' in chart_result:
            logger.warning(f"⚠ Extraction had error: {chart_result['error']}")
        else:
            logger.info("✓ No errors in extraction")

        if chart_result.get('chart_type') and chart_result['chart_type'] != 'unknown':
            logger.info(f"✓ Successfully identified chart type: {chart_result['chart_type']}")
        else:
            logger.warning("⚠ Chart type not identified")

        if chart_result.get('summary'):
            logger.info(f"✓ Summary: {chart_result['summary'][:100]}...")
        else:
            logger.warning("⚠ No summary generated")

    except Exception as e:
        logger.error(f"✗ Chart extraction failed: {e}", exc_info=True)

    logger.info("\n" + "=" * 80)
    logger.info("VLM Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_vlm_extraction()
