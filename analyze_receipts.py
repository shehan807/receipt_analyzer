#!/usr/bin/env python3
"""
Receipt Analyzer CLI - Analyze receipt images/PDFs and generate categorized reports.

Usage:
    python analyze_receipts.py /path/to/reimbursements [-o ./output]
    python analyze_receipts.py /path/to/reimbursements --dry-run

Environment:
    ANTHROPIC_API_KEY - Required. Your Claude API key.
"""

import argparse
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import Category, ReceiptData, ProcessingResult
from claude_client import ClaudeClient
from boundary_detector import BoundaryDetector
from pdf_processor import PDFProcessor
from converter import ImageConverter
from excel_generator import ExcelGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_receipts",
        description="Analyze receipt images/PDFs and generate categorized reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./reimbursements_test
      Process all centers, output to ./output

  %(prog)s ./reimbursements_test -o ./december_results
      Custom output directory

  %(prog)s ./reimbursements_test --dry-run
      Preview files without API calls

Environment:
  ANTHROPIC_API_KEY    Required. Your Claude API key.
        """,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing center folders with receipts",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without processing",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def discover_centers(input_dir: Path) -> dict[str, list[Path]]:
    """Discover center folders and their receipt files.

    Args:
        input_dir: Root directory to search.

    Returns:
        Dictionary mapping center name to list of receipt file paths.
    """
    centers = {}
    valid_extensions = {".pdf", ".png", ".jpg", ".jpeg"}

    for item in input_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            files = []
            for file in item.iterdir():
                if file.is_file() and file.suffix.lower() in valid_extensions:
                    files.append(file)

            if files:
                centers[item.name] = sorted(files)
                logger.info(f"Found center: {item.name} ({len(files)} files)")

    return centers


def process_receipt_file(
    file_path: Path,
    claude_client: ClaudeClient,
    boundary_detector: BoundaryDetector,
    pdf_processor: PDFProcessor,
    converter: ImageConverter,
    temp_dir: Path,
) -> list[ReceiptData]:
    """Process a single receipt file (may contain multiple receipts).

    Args:
        file_path: Path to the receipt file.
        claude_client: Claude client for analysis.
        boundary_detector: Boundary detector for multi-receipt PDFs.
        pdf_processor: PDF processor.
        converter: Image converter.
        temp_dir: Temporary directory for converted files.

    Returns:
        List of ReceiptData extracted from the file.
    """
    receipts = []
    suffix = file_path.suffix.lower()

    logger.info(f"  Processing: {file_path.name}")

    try:
        # Handle images - analyze directly and convert to PDF for combining
        if suffix in (".png", ".jpg", ".jpeg"):
            # Get image bytes for Claude
            image_bytes, media_type = converter.get_image_bytes(file_path)

            # Analyze the receipt
            receipt = claude_client.analyze_receipt_image(
                image_bytes=image_bytes,
                media_type=media_type,
                source_file=file_path.name,
                page_numbers=[1],
                receipt_index=0,
            )
            receipts.append(receipt)

            # Convert to PDF for combining later
            pdf_path = temp_dir / f"{file_path.stem}.pdf"
            converter.convert(file_path, pdf_path)

        # Handle PDFs
        elif suffix == ".pdf":
            # Convert PDF pages to images
            page_images = pdf_processor.pdf_to_images(file_path)

            if len(page_images) == 1:
                # Single page PDF - just analyze it
                img_bytes, page_num = page_images[0]
                receipt = claude_client.analyze_receipt_image(
                    image_bytes=img_bytes,
                    media_type="image/png",
                    source_file=file_path.name,
                    page_numbers=[page_num],
                    receipt_index=0,
                )
                receipts.append(receipt)
            else:
                # Multi-page PDF - detect boundaries
                page_data = [
                    (img_bytes, "image/png", page_num)
                    for img_bytes, page_num in page_images
                ]
                receipt_groups = boundary_detector.detect_boundaries(page_data)

                logger.info(f"    Detected {len(receipt_groups)} receipts in {len(page_images)} pages")

                # Analyze each receipt group
                for idx, page_nums in enumerate(receipt_groups):
                    # For multi-page receipts, combine images or just use first page
                    # (Claude can handle multi-page analysis via first page usually)
                    first_page_idx = page_nums[0] - 1
                    img_bytes = page_images[first_page_idx][0]

                    receipt = claude_client.analyze_receipt_image(
                        image_bytes=img_bytes,
                        media_type="image/png",
                        source_file=file_path.name,
                        page_numbers=page_nums,
                        receipt_index=idx,
                    )
                    receipts.append(receipt)

    except Exception as e:
        logger.error(f"    Error processing {file_path.name}: {e}")
        # Create a placeholder receipt for failed files
        receipts.append(
            ReceiptData(
                source_file=file_path.name,
                store_name="ERROR",
                total=0.0,
                overall_category=Category.QUESTIONABLE,
                confidence="low",
                needs_review=True,
                review_reasons=[f"Processing error: {str(e)}"],
            )
        )

    return receipts


def combine_pdfs_by_category(
    receipts: list[ReceiptData],
    center_path: Path,
    output_dir: Path,
    temp_dir: Path,
    pdf_processor: PDFProcessor,
    converter: ImageConverter,
) -> dict[str, Path]:
    """Combine receipts into category PDFs.

    Args:
        receipts: List of analyzed receipts.
        center_path: Path to center's receipt files.
        output_dir: Output directory for combined PDFs.
        temp_dir: Temporary directory with converted PDFs.
        pdf_processor: PDF processor for merging.
        converter: Image converter.

    Returns:
        Dictionary mapping category to output PDF path.
    """
    # Group receipts by category
    by_category: dict[Category, list[tuple[ReceiptData, Path]]] = {
        Category.SNACK: [],
        Category.SUPPLY: [],
        Category.UNKNOWN: [],
        Category.QUESTIONABLE: [],
    }

    for receipt in receipts:
        # Find the source file (could be in center_path or temp_dir)
        source_name = receipt.source_file
        source_stem = Path(source_name).stem

        # Check for PDF version (either original or converted)
        pdf_path = center_path / source_name
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            # Try temp dir for converted files
            pdf_path = temp_dir / f"{source_stem}.pdf"

        if pdf_path.exists():
            if receipt.overall_category in by_category:
                by_category[receipt.overall_category].append((receipt, pdf_path))
            else:
                by_category[Category.QUESTIONABLE].append((receipt, pdf_path))

    # Create combined PDFs
    output_paths = {}
    category_names = {
        Category.SNACK: "snacks",
        Category.SUPPLY: "supplies",
        Category.UNKNOWN: "unknown",
        Category.QUESTIONABLE: "questionable",
    }

    for category, items in by_category.items():
        if not items:
            continue

        # Get unique PDF paths (some receipts may share source file)
        unique_pdfs = []
        seen = set()
        for receipt, pdf_path in items:
            if str(pdf_path) not in seen:
                seen.add(str(pdf_path))
                unique_pdfs.append(pdf_path)

        output_name = f"{category_names[category]}.pdf"
        output_path = output_dir / output_name

        logger.info(f"    Creating {output_name} ({len(unique_pdfs)} files)")
        pdf_processor.merge_pdfs(unique_pdfs, output_path)
        output_paths[category_names[category]] = output_path

    return output_paths


def process_center(
    center_name: str,
    files: list[Path],
    output_dir: Path,
    claude_client: ClaudeClient,
    boundary_detector: BoundaryDetector,
    pdf_processor: PDFProcessor,
    converter: ImageConverter,
    excel_generator: ExcelGenerator,
) -> ProcessingResult:
    """Process all files in a center.

    Args:
        center_name: Name of the center.
        files: List of receipt files to process.
        output_dir: Output directory for this center.
        claude_client: Claude client.
        boundary_detector: Boundary detector.
        pdf_processor: PDF processor.
        converter: Image converter.
        excel_generator: Excel generator.

    Returns:
        ProcessingResult with all receipts and any errors.
    """
    logger.info(f"\nProcessing center: {center_name}")

    result = ProcessingResult(center_name=center_name)

    # Create output and temp directories
    center_output = output_dir / center_name
    center_output.mkdir(parents=True, exist_ok=True)
    temp_dir = center_output / ".temp"
    temp_dir.mkdir(exist_ok=True)

    center_path = files[0].parent if files else Path()

    try:
        # Process each file
        for file_path in files:
            try:
                file_receipts = process_receipt_file(
                    file_path=file_path,
                    claude_client=claude_client,
                    boundary_detector=boundary_detector,
                    pdf_processor=pdf_processor,
                    converter=converter,
                    temp_dir=temp_dir,
                )
                result.receipts.extend(file_receipts)
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {e}"
                logger.error(f"  {error_msg}")
                result.errors.append(error_msg)

        # Generate Excel summary
        if result.receipts:
            excel_path = center_output / "summary.xlsx"
            logger.info(f"  Generating Excel summary for {center_name}")
            excel_generator.generate(result.receipts, excel_path, center_name=center_name)

            # Combine PDFs by category
            logger.info("  Combining PDFs by category...")
            combine_pdfs_by_category(
                receipts=result.receipts,
                center_path=center_path,
                output_dir=center_output,
                temp_dir=temp_dir,
                pdf_processor=pdf_processor,
                converter=converter,
            )

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def run_dry_run(input_dir: Path) -> None:
    """Preview what would be processed.

    Args:
        input_dir: Input directory to scan.
    """
    print(f"\nDRY RUN - Input directory: {input_dir}\n")

    centers = discover_centers(input_dir)

    if not centers:
        print("No center folders found!")
        return

    total_files = 0
    for center_name, files in sorted(centers.items()):
        print(f"Center: {center_name}")
        for f in files:
            ext = f.suffix.lower()
            print(f"  [{ext[1:].upper():4}] {f.name}")
        print(f"  Total: {len(files)} files\n")
        total_files += len(files)

    print(f"Summary: {len(centers)} centers, {total_files} files")


def print_summary(results: list[ProcessingResult]) -> None:
    """Print processing summary.

    Args:
        results: List of processing results.
    """
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    total_receipts = 0
    total_review = 0
    total_errors = 0

    for result in results:
        receipts = result.receipts
        snacks = sum(1 for r in receipts if r.overall_category == Category.SNACK)
        supplies = sum(1 for r in receipts if r.overall_category == Category.SUPPLY)
        unknown = sum(1 for r in receipts if r.overall_category == Category.UNKNOWN)
        review = sum(1 for r in receipts if r.needs_review)

        print(f"\n{result.center_name}:")
        print(f"  Receipts: {len(receipts)}")
        print(f"  - Snacks: {snacks}")
        print(f"  - Supplies: {supplies}")
        if unknown:
            print(f"  - Unknown (mixed): {unknown}")
        if review:
            print(f"  Needs review: {review}")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")

        total_receipts += len(receipts)
        total_review += review
        total_errors += len(result.errors)

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_receipts} receipts processed")
    if total_review:
        print(f"       {total_review} need review")
    if total_errors:
        print(f"       {total_errors} errors")


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        run_dry_run(args.input_dir)
        return 0

    # Validate API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        print("\nSet it with:", file=sys.stderr)
        print('  export ANTHROPIC_API_KEY="sk-..."', file=sys.stderr)
        return 1

    # Initialize components
    claude_client = ClaudeClient(api_key=api_key)
    boundary_detector = BoundaryDetector(api_key=api_key)
    pdf_processor = PDFProcessor()
    converter = ImageConverter()
    excel_generator = ExcelGenerator()

    # Discover centers
    centers = discover_centers(args.input_dir)

    if not centers:
        print("No center folders with receipt files found!")
        return 1

    # Process each center
    results = []
    for center_name, files in sorted(centers.items()):
        try:
            result = process_center(
                center_name=center_name,
                files=files,
                output_dir=args.output,
                claude_client=claude_client,
                boundary_detector=boundary_detector,
                pdf_processor=pdf_processor,
                converter=converter,
                excel_generator=excel_generator,
            )
            results.append(result)
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
            break
        except Exception as e:
            logger.exception(f"Fatal error processing {center_name}: {e}")
            results.append(
                ProcessingResult(
                    center_name=center_name,
                    errors=[str(e)],
                )
            )

    # Print summary
    print_summary(results)
    print(f"\nOutput directory: {args.output.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
