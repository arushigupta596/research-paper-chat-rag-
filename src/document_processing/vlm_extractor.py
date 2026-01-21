"""
Vision-Language Model for extracting structured data from tables and charts.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

from ..config import config

logger = logging.getLogger(__name__)


class VLMExtractor:
    """
    VLM-based extraction for tables and charts.

    Uses LangChain tools with VLM backend (via OpenRouter) for structured extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize VLM extractor.

        Args:
            api_key: OpenRouter API key (uses config if not provided)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        self.vlm_model = config.VLM_MODEL

        if not self.api_key:
            logger.warning("OpenRouter API key not provided. VLM extraction will fail.")

        logger.info(f"VLM extractor initialized with model: {self.vlm_model}")

    def extract_table(
        self,
        image: Image.Image,
        region_id: str,
        page_num: int
    ) -> Dict[str, Any]:
        """
        Extract structured data from a table image.

        Args:
            image: PIL Image of the table region
            region_id: Unique region identifier
            page_num: Page number

        Returns:
            Dictionary with structured table data
        """
        logger.debug(f"Extracting table from {region_id}")

        # Create prompt for table extraction
        prompt = """Analyze this table image and extract structured information.

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
Do not use markdown code blocks. Just return the raw JSON.

Return this exact structure:
{
    "headers": ["col1", "col2"],
    "rows": [["val1", "val2"], ["val3", "val4"]],
    "units": "description of units if any or empty string",
    "footnotes": "any footnotes or empty string",
    "summary": "brief description of what the table shows"
}

Extract all visible data. If you cannot see the table clearly, return empty arrays for headers and rows.
Respond with JSON only, no other text."""

        try:
            result = self._call_vlm(image, prompt)

            # Clean response (remove markdown code blocks, extra text)
            cleaned_result = self._clean_json_response(result)

            # Parse JSON response
            table_data = json.loads(cleaned_result)

            # Validate required fields
            if 'headers' not in table_data:
                table_data['headers'] = []
            if 'rows' not in table_data:
                table_data['rows'] = []
            if 'summary' not in table_data:
                table_data['summary'] = ''

            # Add metadata
            table_data['region_id'] = region_id
            table_data['page_num'] = page_num
            table_data['type'] = 'table'

            return table_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {region_id}: {e}")
            logger.debug(f"Raw response: {result[:200] if 'result' in locals() else 'N/A'}")
            return {
                'region_id': region_id,
                'page_num': page_num,
                'type': 'table',
                'error': f'JSON parse error: {str(e)}',
                'headers': [],
                'rows': [],
                'summary': 'Table extraction failed - invalid JSON response'
            }
        except Exception as e:
            logger.error(f"Failed to extract table from {region_id}: {e}")
            return {
                'region_id': region_id,
                'page_num': page_num,
                'type': 'table',
                'error': str(e),
                'headers': [],
                'rows': [],
                'summary': 'Table extraction failed'
            }

    def extract_chart(
        self,
        image: Image.Image,
        region_id: str,
        page_num: int
    ) -> Dict[str, Any]:
        """
        Extract structured data from a chart/figure image.

        Args:
            image: PIL Image of the chart region
            region_id: Unique region identifier
            page_num: Page number

        Returns:
            Dictionary with structured chart data
        """
        logger.debug(f"Extracting chart from {region_id}")

        # Create prompt for chart extraction
        prompt = """Analyze this chart/figure and extract key information.

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
Do not use markdown code blocks. Just return the raw JSON.

Return this exact structure:
{
    "chart_type": "bar/line/scatter/pie/diagram/other",
    "title": "chart title if visible or empty string",
    "x_axis": {"label": "x-axis label", "values": ["val1", "val2"]},
    "y_axis": {"label": "y-axis label", "range": "min-max"},
    "data_series": [{"name": "series1", "description": "what it shows"}],
    "key_insights": ["insight1", "insight2"],
    "trends": "description of visible trends",
    "anomalies": "any notable anomalies or empty string",
    "summary": "comprehensive description of what the chart shows"
}

If you cannot clearly see certain elements, use empty strings or empty arrays.
Respond with JSON only, no other text."""

        try:
            result = self._call_vlm(image, prompt)

            # Clean response (remove markdown code blocks, extra text)
            cleaned_result = self._clean_json_response(result)

            # Parse JSON response
            chart_data = json.loads(cleaned_result)

            # Validate required fields
            if 'chart_type' not in chart_data:
                chart_data['chart_type'] = 'unknown'
            if 'summary' not in chart_data:
                chart_data['summary'] = ''

            # Add metadata
            chart_data['region_id'] = region_id
            chart_data['page_num'] = page_num
            chart_data['type'] = 'chart'

            return chart_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {region_id}: {e}")
            logger.debug(f"Raw response: {result[:200] if 'result' in locals() else 'N/A'}")
            return {
                'region_id': region_id,
                'page_num': page_num,
                'type': 'chart',
                'error': f'JSON parse error: {str(e)}',
                'chart_type': 'unknown',
                'summary': 'Chart extraction failed - invalid JSON response'
            }
        except Exception as e:
            logger.error(f"Failed to extract chart from {region_id}: {e}")
            return {
                'region_id': region_id,
                'page_num': page_num,
                'type': 'chart',
                'error': str(e),
                'chart_type': 'unknown',
                'summary': 'Chart extraction failed'
            }

    def _clean_json_response(self, response: str) -> str:
        """
        Clean VLM response to extract valid JSON.

        Removes markdown code blocks, extra text, and whitespace.

        Args:
            response: Raw VLM response

        Returns:
            Cleaned JSON string
        """
        import re

        # Remove markdown code blocks
        if '```json' in response:
            # Extract content between ```json and ```
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif '```' in response:
            # Extract content between ``` and ```
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)

        # Find first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            response = response[first_brace:last_brace + 1]

        return response.strip()

    def _call_vlm(self, image: Image.Image, prompt: str, max_retries: int = 3) -> str:
        """
        Call VLM API with image and prompt, with retry logic for rate limits.

        Args:
            image: PIL Image
            prompt: Text prompt
            max_retries: Maximum number of retry attempts

        Returns:
            VLM response text
        """
        import time

        for attempt in range(max_retries):
            try:
                # Import OpenAI client here to avoid dependency issues
                from openai import OpenAI

                # Encode image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Initialize client
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )

                # Call VLM
                response = client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )

                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)

                # Check for rate limit error (429)
                if '429' in error_str or 'rate limit' in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                        logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                logger.error(f"VLM API call failed: {e}")
                raise

        raise Exception(f"Failed after {max_retries} retries")

    def process_regions(
        self,
        regions: List[Dict[str, Any]],
        images: Dict[int, Image.Image]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple table/chart regions.

        Args:
            regions: List of layout regions (filtered for tables/figures)
            images: Dictionary mapping page_num to PIL Image

        Returns:
            Dictionary mapping region_id to extracted structured data
        """
        extracted_data = {}

        for region in regions:
            region_id = region['region_id']
            region_type = region['region_type']
            page_num = region['page_num']
            bbox = region['bbox']

            if page_num not in images:
                logger.warning(f"Image not found for page {page_num}")
                continue

            # Crop region from image
            image = images[page_num]
            x1, y1, x2, y2 = map(int, bbox)
            cropped = image.crop((x1, y1, x2, y2))

            # Extract based on type
            if region_type == 'table':
                data = self.extract_table(cropped, region_id, page_num)
            elif region_type == 'figure':
                data = self.extract_chart(cropped, region_id, page_num)
            else:
                continue

            extracted_data[region_id] = data

        logger.info(f"Processed {len(extracted_data)} table/chart regions")
        return extracted_data
