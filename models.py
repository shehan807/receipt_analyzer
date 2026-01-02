"""Data models for receipt analysis."""

from enum import Enum
from typing import Optional, Union
import datetime
from pydantic import BaseModel, Field


class Category(str, Enum):
    """Receipt/item category."""
    SNACK = "snack"
    SUPPLY = "supply"
    QUESTIONABLE = "questionable"
    UNKNOWN = "unknown"  # For mixed-category receipts


class Confidence(str, Enum):
    """Confidence level for extracted data."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LineItem(BaseModel):
    """A single item from a receipt."""
    name: str
    price: float
    quantity: int = 1
    category: Category
    confidence: Confidence
    category_reasoning: Optional[str] = None


class ReceiptData(BaseModel):
    """Extracted data from a single receipt."""
    # Source tracking
    source_file: str
    page_numbers: list[int] = Field(default_factory=list)  # Pages this receipt spans
    receipt_index: int = 0  # Index within source file if multiple receipts

    # Core receipt info
    store_name: str
    store_address: Optional[str] = None
    receipt_date: Union[datetime.date, None] = None  # Renamed to avoid conflict
    date_raw: Optional[str] = None  # Keep original if parsing fails

    # Financials
    subtotal: Optional[float] = None
    tax_paid: bool = False
    tax_amount: Optional[float] = None
    delivery_cost: Optional[float] = None  # Delivery/shipping fee if applicable
    total: float = 0.0  # Default to 0 if not found

    # Line items
    items: list[LineItem] = Field(default_factory=list)

    # Classification
    overall_category: Category
    confidence: Confidence
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    # Additional metadata
    payment_method: Optional[str] = None  # e.g., "VISA", "CASH", "MASTERCARD"
    card_last_four: Optional[str] = None  # Last 4 digits if card payment
    notes: Optional[str] = None


class ProcessingResult(BaseModel):
    """Result of processing a single center."""
    center_name: str
    receipts: list[ReceiptData] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BoundaryInfo(BaseModel):
    """Information about receipt boundaries in a PDF."""
    page_num: int
    is_receipt_start: bool
    is_continuation: bool = False
    has_total: bool = False
    store_name: Optional[str] = None
    date_raw: Optional[str] = None
    confidence: Confidence = Confidence.MEDIUM
