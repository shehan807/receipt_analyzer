"""Detect receipt boundaries in multi-page PDFs."""

import base64
import json
import re
import logging
import io
import time
from typing import Optional

import anthropic
from PIL import Image, ImageOps

from models import BoundaryInfo, Confidence

logger = logging.getLogger(__name__)

# Maximum image size for Claude API (5MB)
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
MAX_IMAGE_DIMENSION = 7500  # Claude limit is 8000, leave margin

# Increase PIL's decompression bomb limit
Image.MAX_IMAGE_PIXELS = 300000000  # 300 megapixels

BOUNDARY_DETECTION_PROMPT = """Analyze this receipt image page and answer these questions.

{context}

Look for these indicators:
1. Is there a store HEADER/LOGO at the TOP of the page? (indicates new receipt)
2. What store is this from?
3. What is the TRANSACTION/PURCHASE date? (IGNORE "Return by", "Eligible through", "Expires" dates - look for "Order placed" or date near top)
4. Does this page contain a TOTAL line (indicating receipt end)?
5. Does this appear to be a CONTINUATION of a previous receipt (no header, starts mid-receipt)?
6. Is this a completely new/different receipt from the previous one?

Respond ONLY with valid JSON (no markdown, no explanation):
{{
    "is_receipt_start": true,
    "is_continuation": false,
    "has_total": true,
    "store_name": "Store Name or null",
    "date_raw": "MM/DD/YY or null",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation of your determination"
}}

Important:
- is_receipt_start=true if this page begins a NEW receipt (has header/logo at top)
- is_continuation=true if this continues a previous receipt (no header, starts with items)
- A single-page receipt has both is_receipt_start=true AND has_total=true"""


class BoundaryDetector:
    """Detect receipt boundaries in multi-page PDFs."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """Initialize boundary detector.

        Args:
            api_key: Anthropic API key.
            model: Model to use for analysis.
            max_retries: Maximum number of retries for rate limiting.
            base_delay: Base delay in seconds for exponential backoff.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _call_api_with_retry(self, messages: list, max_tokens: int) -> anthropic.types.Message:
        """Call Claude API with retry logic for rate limiting.

        Args:
            messages: Messages to send.
            max_tokens: Maximum tokens in response.

        Returns:
            API response.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
            except anthropic.RateLimitError as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {delay}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                raise

        raise last_error

    def _compress_image(self, image_bytes: bytes) -> tuple[bytes, str]:
        """Compress image if it exceeds the maximum size or dimensions.

        Args:
            image_bytes: Original image bytes.

        Returns:
            Tuple of (compressed_bytes, media_type).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            original_size = (img.width, img.height)

            # Apply EXIF orientation correction (handles rotated phone photos)
            img = ImageOps.exif_transpose(img)

            # Check if EXIF correction changed the image (dimensions changed = was rotated)
            exif_rotated = (img.width, img.height) != original_size

            # Check if we need to process
            needs_resize = max(img.width, img.height) > MAX_IMAGE_DIMENSION
            needs_compress = len(image_bytes) > MAX_IMAGE_SIZE

            if not needs_resize and not needs_compress and not exif_rotated:
                # Detect media type
                if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                    return image_bytes, "image/png"
                return image_bytes, "image/jpeg"

            # If only EXIF rotation was needed, save with original quality
            if exif_rotated and not needs_resize and not needs_compress:
                logger.info(f"Applying EXIF rotation: {original_size} -> {img.width}x{img.height}")
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95, optimize=True)
                return buffer.getvalue(), "image/jpeg"

            logger.info(f"Processing boundary image: {img.width}x{img.height}, {len(image_bytes) / 1024 / 1024:.2f}MB")

            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # First resize if dimensions too large
            if needs_resize:
                scale = MAX_IMAGE_DIMENSION / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size[0]}x{new_size[1]}")

            # Try different quality levels
            for quality in [85, 70, 55, 40, 25]:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                compressed = buffer.getvalue()

                if len(compressed) <= MAX_IMAGE_SIZE:
                    logger.info(f"Compressed to {len(compressed) / 1024 / 1024:.2f}MB")
                    return compressed, "image/jpeg"

            # Resize further if still too large
            for scale in [0.7, 0.5, 0.3]:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                resized.save(buffer, format='JPEG', quality=60, optimize=True)
                compressed = buffer.getvalue()

                if len(compressed) <= MAX_IMAGE_SIZE:
                    logger.info(f"Resized to {new_size} and compressed to {len(compressed) / 1024 / 1024:.2f}MB")
                    return compressed, "image/jpeg"

            # Last resort
            new_size = (int(img.width * 0.2), int(img.height * 0.2))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG', quality=50, optimize=True)
            compressed = buffer.getvalue()
            logger.warning(f"Heavily compressed to {len(compressed) / 1024 / 1024:.2f}MB")
            return compressed, "image/jpeg"

        except Exception as e:
            logger.error(f"Failed to compress image: {e}")
            return image_bytes, "image/png"

    def detect_boundaries(
        self,
        page_images: list[tuple[bytes, str, int]],
    ) -> list[list[int]]:
        """Detect receipt boundaries in a multi-page document.

        Args:
            page_images: List of (image_bytes, media_type, page_num) tuples.

        Returns:
            List of receipt groups, each containing page numbers.
            Example: [[1], [2], [3, 4], [5]] means 4 receipts,
                     where receipt 3 spans pages 3-4.
        """
        if not page_images:
            return []

        # If single page, it's one receipt
        if len(page_images) == 1:
            return [[page_images[0][2]]]

        # Analyze each page
        boundary_info = self._analyze_all_pages(page_images)

        # Build receipt groups
        receipt_groups = []
        current_group = []

        for info in boundary_info:
            if info.is_receipt_start and current_group:
                # Save previous group and start new one
                receipt_groups.append(current_group)
                current_group = [info.page_num]
            else:
                current_group.append(info.page_num)

        # Don't forget the last group
        if current_group:
            receipt_groups.append(current_group)

        logger.info(f"Detected {len(receipt_groups)} receipts across {len(page_images)} pages")
        return receipt_groups

    def _analyze_all_pages(
        self,
        page_images: list[tuple[bytes, str, int]],
    ) -> list[BoundaryInfo]:
        """Analyze each page to determine receipt boundaries.

        Args:
            page_images: List of (image_bytes, media_type, page_num) tuples.

        Returns:
            List of BoundaryInfo for each page.
        """
        results = []
        prev_store = None
        prev_date = None

        for img_bytes, media_type, page_num in page_images:
            # Build context from previous pages
            context = ""
            if prev_store:
                context += f"\nPrevious receipt was from: {prev_store}"
            if prev_date:
                context += f"\nPrevious receipt date: {prev_date}"

            if page_num == 1:
                context = "\nThis is the FIRST page of the document."

            prompt = BOUNDARY_DETECTION_PROMPT.format(context=context)

            try:
                info = self._analyze_single_page(img_bytes, media_type, page_num, prompt)
                results.append(info)

                # Update context for next page
                if info.store_name:
                    prev_store = info.store_name
                if info.date_raw:
                    prev_date = info.date_raw

            except Exception as e:
                logger.error(f"Failed to analyze page {page_num}: {e}")
                # Default to assuming new receipt on error
                results.append(
                    BoundaryInfo(
                        page_num=page_num,
                        is_receipt_start=True,
                        confidence=Confidence.LOW,
                    )
                )

        return results

    def _analyze_single_page(
        self,
        image_bytes: bytes,
        media_type: str,
        page_num: int,
        prompt: str,
    ) -> BoundaryInfo:
        """Analyze a single page for boundary detection.

        Args:
            image_bytes: Raw image bytes.
            media_type: MIME type of the image.
            page_num: Page number.
            prompt: Analysis prompt.

        Returns:
            BoundaryInfo for this page.
        """
        # Compress image if needed
        image_bytes, media_type = self._compress_image(image_bytes)

        base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

        response = self._call_api_with_retry(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
        )

        # Validate response has content
        if not response.content or not hasattr(response.content[0], 'text'):
            logger.warning(f"Empty response from API for page {page_num}")
            return BoundaryInfo(
                page_num=page_num,
                is_receipt_start=True,
                confidence=Confidence.LOW,
            )

        raw_text = response.content[0].text
        if not raw_text:
            logger.warning(f"Empty response text for page {page_num}")
            return BoundaryInfo(
                page_num=page_num,
                is_receipt_start=True,
                confidence=Confidence.LOW,
            )

        parsed = self._parse_boundary_response(raw_text)

        # Safely parse confidence
        confidence_str = parsed.get("confidence", "medium")
        try:
            confidence = Confidence(confidence_str.lower() if isinstance(confidence_str, str) else "medium")
        except ValueError:
            confidence = Confidence.MEDIUM

        return BoundaryInfo(
            page_num=page_num,
            is_receipt_start=parsed.get("is_receipt_start", True),
            is_continuation=parsed.get("is_continuation", False),
            has_total=parsed.get("has_total", False),
            store_name=parsed.get("store_name"),
            date_raw=parsed.get("date_raw"),
            confidence=confidence,
        )

    def _parse_boundary_response(self, response_text: str) -> dict:
        """Parse Claude's boundary detection response.

        Args:
            response_text: Raw response text.

        Returns:
            Parsed JSON dictionary.
        """
        # First try to parse the entire response as JSON
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Extract JSON by finding balanced braces
        json_str = self._extract_json_object(response_text)
        if not json_str:
            logger.warning(f"No JSON in boundary response: {response_text[:100]}")
            return {"is_receipt_start": True, "confidence": "low"}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix trailing commas
            cleaned = re.sub(r",\s*}", "}", json_str)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse boundary JSON: {e}")
                return {"is_receipt_start": True, "confidence": "low"}

    def _extract_json_object(self, text: str) -> str:
        """Extract a JSON object from text by finding balanced braces.

        Args:
            text: Text containing JSON.

        Returns:
            Extracted JSON string or empty string if not found.
        """
        start_idx = text.find('{')
        if start_idx == -1:
            return ""

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i + 1]

        return ""
