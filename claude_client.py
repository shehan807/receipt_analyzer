"""Claude API client for receipt analysis."""

import base64
import json
import re
import time
import logging
import io
from typing import Optional
from datetime import datetime

import anthropic
from PIL import Image, ImageOps

from models import Category, Confidence, LineItem, ReceiptData

logger = logging.getLogger(__name__)

# Maximum image size for Claude API (5MB)
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
MAX_IMAGE_DIMENSION = 7500  # Claude limit is 8000, leave some margin

# Increase PIL's decompression bomb limit for large receipt scans
Image.MAX_IMAGE_PIXELS = 300000000  # 300 megapixels

# Default categorization rules - can be customized by users
DEFAULT_CATEGORIZATION_RULES = """- SNACK: Food items for children (chips, cookies, drinks, pizza, chicken, fruit, croissants, donuts, candy, etc.)
- SUPPLY: Non-food items (paper plates, napkins, tape, craft supplies, paint, markers, planters, decorations, cleaning supplies, office supplies)
- QUESTIONABLE: Items that could be either or are unclear

IMPORTANT CATEGORIZATION RULES:
1. Food items = SNACK (even bulk food like croissants, pizza orders)
2. Craft supplies, office supplies, tape, decorations = SUPPLY
3. Unclear items = QUESTIONABLE
4. If store is known for food (Pizza Hut, Popeyes, Dunkin, McDonald's, Chick-fil-A) = default SNACK
5. If store is craft-focused (Michaels, JOANN) = likely SUPPLY
6. Dollar stores (Dollar Tree, Dollar General) can have either - categorize each item"""


def build_analysis_prompt(categorization_rules: str = None) -> str:
    """Build the full analysis prompt with custom categorization rules.

    Args:
        categorization_rules: Custom rules or None for defaults.

    Returns:
        Full analysis prompt string.
    """
    rules = categorization_rules or DEFAULT_CATEGORIZATION_RULES

    # Use raw string with explicit braces to avoid f-string issues with JSON
    return f"""You are a receipt analyzer for a childcare center reimbursement system.

Analyze this receipt image and extract all information. The items should be categorized as:
{rules}

DATE EXTRACTION RULES:
- Extract the TRANSACTION/PURCHASE/ORDER date - when the purchase was made
- IGNORE these dates: "Return by", "Eligible through", "Expires", "Ship by", "Delivery date"
- Look for: "Order placed", "Transaction date", "Date:", or the date near the top of the receipt
- For online orders, use "Order placed" date, NOT return eligibility date

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{{
    "store_name": "string",
    "store_address": "string or null",
    "date": "YYYY-MM-DD or null if unclear (MUST be transaction/purchase date, not return deadline)",
    "date_raw": "original transaction date text from receipt",

    "items": [
        {{
            "name": "item name as shown",
            "quantity": 1,
            "price": 1.25,
            "category": "snack|supply|questionable",
            "category_reasoning": "brief explanation",
            "confidence": "high|medium|low"
        }}
    ],

    "subtotal": 0.00,
    "tax_paid": true,
    "tax_amount": 0.00,
    "delivery_cost": 0.00,
    "total": 0.00,

    "payment_method": "VISA|MASTERCARD|AMEX|DISCOVER|DEBIT|CASH|CHECK|etc or null",
    "card_last_four": "1234 or null if not visible or not a card",
    "confidence": "high|medium|low",
    "needs_review": false,
    "review_reasons": [],

    "notes": "any special observations or null"
}}

FLAG FOR REVIEW (set needs_review=true and add to review_reasons) if:
- Handwritten receipt with unclear amounts
- Total doesn't match subtotal + tax
- Items are ambiguous between snack/supply
- Date is unclear or missing
- Low confidence on any major field
- Receipt is partially cut off or blurry
- Store name cannot be determined"""


class ClaudeClient:
    """Client for Claude API with receipt analysis capabilities."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        base_delay: float = 1.0,
        custom_categorization_rules: str = None,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key.
            model: Model to use for analysis.
            max_retries: Maximum number of retries for rate limiting.
            base_delay: Base delay in seconds for exponential backoff.
            custom_categorization_rules: Custom rules for item categorization, or None for defaults.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.analysis_prompt = build_analysis_prompt(custom_categorization_rules)

    def _detect_media_type(self, image_bytes: bytes) -> str:
        """Detect actual media type from image bytes using magic numbers.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Detected media type string.
        """
        # Check magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            # Default to JPEG
            return "image/jpeg"

    def _compress_image(self, image_bytes: bytes, media_type: str) -> tuple[bytes, str]:
        """Compress image if it exceeds the maximum size or dimensions.

        Args:
            image_bytes: Original image bytes.
            media_type: Original media type.

        Returns:
            Tuple of (compressed_bytes, media_type).
        """
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(image_bytes))
            original_size = (img.width, img.height)

            # Apply EXIF orientation correction (handles rotated phone photos)
            img = ImageOps.exif_transpose(img)

            # Check if EXIF correction changed the image (dimensions changed = was rotated)
            exif_rotated = (img.width, img.height) != original_size

            # Check if we need to process (size or dimensions)
            needs_resize = max(img.width, img.height) > MAX_IMAGE_DIMENSION
            needs_compress = len(image_bytes) > MAX_IMAGE_SIZE

            if not needs_resize and not needs_compress and not exif_rotated:
                return image_bytes, media_type

            # If only EXIF rotation was needed, save with original quality
            if exif_rotated and not needs_resize and not needs_compress:
                logger.info(f"Applying EXIF rotation correction: {original_size} -> {img.width}x{img.height}")
                # Convert to RGB if necessary for JPEG
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

            logger.info(f"Processing image: {img.width}x{img.height}, {len(image_bytes) / 1024 / 1024:.2f}MB")

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

            # First, resize if dimensions exceed limit
            if needs_resize:
                scale = MAX_IMAGE_DIMENSION / max(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size[0]}x{new_size[1]}")

            # Try different quality levels
            for quality in [85, 70, 55, 40, 25, 15]:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                compressed = buffer.getvalue()

                if len(compressed) <= MAX_IMAGE_SIZE:
                    logger.info(f"Compressed to {len(compressed) / 1024 / 1024:.2f}MB at quality {quality}")
                    return compressed, "image/jpeg"

            # If still too large, resize further
            for scale in [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                resized.save(buffer, format='JPEG', quality=60, optimize=True)
                compressed = buffer.getvalue()

                if len(compressed) <= MAX_IMAGE_SIZE:
                    logger.info(f"Resized to {new_size} and compressed to {len(compressed) / 1024 / 1024:.2f}MB")
                    return compressed, "image/jpeg"

            # Last resort - very small and low quality
            new_size = (int(img.width * 0.15), int(img.height * 0.15))
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format='JPEG', quality=40, optimize=True)
            compressed = buffer.getvalue()
            logger.warning(f"Heavily compressed to {len(compressed) / 1024 / 1024:.2f}MB at {new_size}")
            return compressed, "image/jpeg"

        except Exception as e:
            logger.error(f"Failed to compress image: {e}")
            return image_bytes, media_type

    def analyze_receipt_image(
        self,
        image_bytes: bytes,
        media_type: str,
        source_file: str,
        page_numbers: Optional[list[int]] = None,
        receipt_index: int = 0,
    ) -> ReceiptData:
        """Analyze a receipt image using Claude Vision.

        Args:
            image_bytes: Raw image bytes.
            media_type: MIME type of the image (e.g., "image/png").
            source_file: Name of the source file.
            page_numbers: Page numbers this receipt spans (for multi-page).
            receipt_index: Index of this receipt within the source file.

        Returns:
            ReceiptData with extracted information.
        """
        # Detect actual media type from file contents
        actual_media_type = self._detect_media_type(image_bytes)
        if actual_media_type != media_type:
            logger.debug(f"Corrected media type from {media_type} to {actual_media_type}")
            media_type = actual_media_type

        # Compress if needed
        image_bytes, media_type = self._compress_image(image_bytes, media_type)

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
                        {"type": "text", "text": self.analysis_prompt},
                    ],
                }
            ],
            max_tokens=4096,
        )

        # Validate response has content
        if not response.content or not hasattr(response.content[0], 'text'):
            raise ValueError(f"Empty or invalid response from Claude API for {source_file}")

        raw_text = response.content[0].text
        if not raw_text:
            raise ValueError(f"Empty response text from Claude API for {source_file}")

        parsed = self._parse_response(raw_text)

        return self._build_receipt_data(
            parsed,
            source_file=source_file,
            page_numbers=page_numbers or [],
            receipt_index=receipt_index,
        )

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
                delay = self.base_delay * (2**attempt)
                logger.warning(f"Rate limited, waiting {delay}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                raise

        raise last_error

    def _parse_response(self, response_text: str) -> dict:
        """Parse Claude's JSON response with error handling.

        Args:
            response_text: Raw response text from Claude.

        Returns:
            Parsed JSON dictionary.
        """
        # First, try to parse the entire response as JSON (ideal case)
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Find JSON by locating balanced braces
        json_str = self._extract_json_object(response_text)
        if not json_str:
            raise ValueError(f"No valid JSON found in response: {response_text[:200]}...")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix trailing commas (common LLM issue)
            cleaned = re.sub(r",\s*}", "}", json_str)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {e}. Content: {json_str[:200]}...")

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

    def _build_receipt_data(
        self,
        parsed: dict,
        source_file: str,
        page_numbers: list[int],
        receipt_index: int,
    ) -> ReceiptData:
        """Build ReceiptData from parsed JSON.

        Args:
            parsed: Parsed JSON from Claude.
            source_file: Source file name.
            page_numbers: Page numbers this receipt spans.
            receipt_index: Index within source file.

        Returns:
            ReceiptData instance.
        """
        # Parse items
        items = []
        for item_data in parsed.get("items", []):
            try:
                # Safely extract price
                price_val = item_data.get("price")
                if price_val is None:
                    price_val = 0.0
                else:
                    try:
                        price_val = float(price_val)
                    except (ValueError, TypeError):
                        price_val = 0.0

                # Safely extract quantity
                qty_val = item_data.get("quantity", 1)
                if qty_val is None:
                    qty_val = 1
                else:
                    try:
                        qty_val = int(qty_val)
                    except (ValueError, TypeError):
                        qty_val = 1

                # Safely parse category enum
                category_str = item_data.get("category", "questionable") or "questionable"
                try:
                    category = Category(category_str.lower())
                except ValueError:
                    logger.warning(f"Unknown category '{category_str}' for item, defaulting to questionable")
                    category = Category.QUESTIONABLE

                # Safely parse confidence enum
                confidence_str = item_data.get("confidence", "medium") or "medium"
                try:
                    confidence = Confidence(confidence_str.lower())
                except ValueError:
                    logger.warning(f"Unknown confidence '{confidence_str}' for item, defaulting to medium")
                    confidence = Confidence.MEDIUM

                items.append(
                    LineItem(
                        name=item_data.get("name", "Unknown") or "Unknown",
                        price=price_val,
                        quantity=qty_val,
                        category=category,
                        confidence=confidence,
                        category_reasoning=item_data.get("category_reasoning"),
                    )
                )
            except (ValueError, KeyError, AttributeError) as e:
                logger.warning(f"Failed to parse item: {item_data}, error: {e}")

        # Parse date
        parsed_date = None
        date_raw = parsed.get("date_raw")
        if parsed.get("date"):
            try:
                parsed_date = datetime.strptime(parsed["date"], "%Y-%m-%d").date()
            except ValueError:
                pass

        # Determine overall category based on items
        overall_category = self._determine_overall_category(items)

        # Update needs_review if mixed categories
        needs_review = parsed.get("needs_review", False)
        review_reasons = parsed.get("review_reasons", [])

        if overall_category == Category.UNKNOWN:
            needs_review = True
            if "Mixed categories - internal issue" not in review_reasons:
                review_reasons.append("Mixed categories - internal issue")

        # Handle None total
        total_value = parsed.get("total")
        if total_value is None:
            total_value = 0.0
            needs_review = True
            if "Total not found" not in review_reasons:
                review_reasons.append("Total not found")
        else:
            try:
                total_value = float(total_value)
            except (ValueError, TypeError):
                total_value = 0.0
                needs_review = True
                if "Invalid total value" not in review_reasons:
                    review_reasons.append("Invalid total value")

        # Handle tax_paid - must handle string "false" correctly
        tax_paid_val = parsed.get("tax_paid")
        if isinstance(tax_paid_val, bool):
            tax_paid = tax_paid_val
        elif isinstance(tax_paid_val, str):
            tax_paid = tax_paid_val.lower() == "true"
        else:
            tax_paid = False

        # Safely parse overall confidence
        confidence_str = parsed.get("confidence", "medium") or "medium"
        try:
            overall_confidence = Confidence(confidence_str.lower() if isinstance(confidence_str, str) else "medium")
        except ValueError:
            overall_confidence = Confidence.MEDIUM

        return ReceiptData(
            source_file=source_file,
            page_numbers=page_numbers,
            receipt_index=receipt_index,
            store_name=parsed.get("store_name", "Unknown") or "Unknown",
            store_address=parsed.get("store_address"),
            receipt_date=parsed_date,
            date_raw=date_raw,
            subtotal=parsed.get("subtotal"),
            tax_paid=tax_paid,
            tax_amount=parsed.get("tax_amount"),
            delivery_cost=parsed.get("delivery_cost"),
            total=total_value,
            items=items,
            overall_category=overall_category,
            confidence=overall_confidence,
            needs_review=needs_review,
            review_reasons=review_reasons,
            payment_method=parsed.get("payment_method"),
            card_last_four=parsed.get("card_last_four"),
            notes=parsed.get("notes"),
        )

    def _determine_overall_category(self, items: list[LineItem]) -> Category:
        """Determine overall receipt category based on items.

        Args:
            items: List of line items.

        Returns:
            Overall category for the receipt.
        """
        if not items:
            return Category.QUESTIONABLE

        category_counts = {
            Category.SNACK: 0,
            Category.SUPPLY: 0,
            Category.QUESTIONABLE: 0,
        }

        for item in items:
            if item.category in category_counts:
                category_counts[item.category] += 1

        snacks = category_counts[Category.SNACK]
        supplies = category_counts[Category.SUPPLY]
        questionable = category_counts[Category.QUESTIONABLE]

        # If all items are one category (or questionable), assign that
        if snacks > 0 and supplies == 0:
            return Category.SNACK
        elif supplies > 0 and snacks == 0:
            return Category.SUPPLY
        elif snacks > 0 and supplies > 0:
            # Mixed - this is the rare case that indicates internal issue
            return Category.UNKNOWN
        else:
            return Category.QUESTIONABLE
