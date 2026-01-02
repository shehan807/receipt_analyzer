"""PDF processing utilities using PyMuPDF."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Iterator


class PDFProcessor:
    """Handle PDF reading, splitting, and merging."""

    def __init__(self, dpi: int = 200):
        """Initialize PDF processor.

        Args:
            dpi: Resolution for rendering PDF pages as images.
        """
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: Path) -> list[tuple[bytes, int]]:
        """Convert PDF pages to PNG images.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of (image_bytes, page_number) tuples. Page numbers are 1-indexed.
        """
        doc = fitz.open(pdf_path)
        images = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render at specified DPI
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")
                images.append((img_bytes, page_num + 1))
        finally:
            doc.close()

        return images

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of pages.
        """
        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()

    def merge_pdfs(self, pdf_paths: list[Path], output_path: Path) -> None:
        """Merge multiple PDFs into one.

        Args:
            pdf_paths: List of PDF paths to merge.
            output_path: Path for the merged output PDF.
        """
        if not pdf_paths:
            return

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged = fitz.open()
        try:
            for pdf_path in pdf_paths:
                doc = fitz.open(pdf_path)
                try:
                    merged.insert_pdf(doc)
                finally:
                    doc.close()

            merged.save(output_path)
        finally:
            merged.close()

    def extract_pages(
        self,
        pdf_path: Path,
        page_ranges: list[tuple[int, int]],
        output_dir: Path,
    ) -> list[Path]:
        """Extract page ranges into separate PDFs.

        Args:
            pdf_path: Source PDF path.
            page_ranges: List of (start, end) page numbers (1-indexed, inclusive).
            output_dir: Directory to save extracted PDFs.

        Returns:
            List of paths to extracted PDFs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        output_paths = []

        try:
            for i, (start, end) in enumerate(page_ranges):
                new_doc = fitz.open()
                try:
                    # fitz uses 0-indexed pages
                    new_doc.insert_pdf(doc, from_page=start - 1, to_page=end - 1)

                    output_path = output_dir / f"{pdf_path.stem}_receipt_{i + 1}.pdf"
                    new_doc.save(output_path)
                    output_paths.append(output_path)
                finally:
                    new_doc.close()
        finally:
            doc.close()

        return output_paths

    def extract_single_page(self, pdf_path: Path, page_num: int, output_path: Path) -> Path:
        """Extract a single page to a new PDF.

        Args:
            pdf_path: Source PDF path.
            page_num: Page number to extract (1-indexed).
            output_path: Path for the output PDF.

        Returns:
            Path to the extracted PDF.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        try:
            new_doc = fitz.open()
            try:
                new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                new_doc.save(output_path)
            finally:
                new_doc.close()
        finally:
            doc.close()

        return output_path
