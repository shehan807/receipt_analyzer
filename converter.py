"""Image to PDF conversion utilities for macOS."""

import subprocess
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageConverter:
    """Convert images to PDFs using macOS native tools with Pillow fallback."""

    @staticmethod
    def image_to_pdf_sips(image_path: Path, output_path: Path) -> bool:
        """Convert image to PDF using macOS sips command.

        Args:
            image_path: Path to the source image.
            output_path: Path for the output PDF.

        Returns:
            True if conversion succeeded, False otherwise.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["sips", "-s", "format", "pdf", str(image_path), "--out", str(output_path)],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.debug(f"sips conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except FileNotFoundError:
            logger.debug("sips command not found (not on macOS?)")
            return False

    @staticmethod
    def image_to_pdf_pillow(image_path: Path, output_path: Path) -> bool:
        """Convert image to PDF using Pillow.

        Args:
            image_path: Path to the source image.
            output_path: Path for the output PDF.

        Returns:
            True if conversion succeeded, False otherwise.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path)

            # Convert to RGB if necessary (PDF doesn't support RGBA/LA/P)
            if img.mode in ("RGBA", "LA"):
                # Create white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Paste image using alpha channel as mask
                if img.mode == "LA":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == "P":
                # Palette mode - convert through RGBA to handle transparency
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.save(output_path, "PDF", resolution=150.0)
            return True
        except Exception as e:
            logger.error(f"Pillow conversion failed: {e}")
            return False

    def convert(self, image_path: Path, output_path: Path) -> bool:
        """Convert image to PDF, trying sips first, then Pillow fallback.

        Args:
            image_path: Path to the source image.
            output_path: Path for the output PDF.

        Returns:
            True if conversion succeeded, False otherwise.
        """
        # Try macOS sips first (produces better quality)
        if self.image_to_pdf_sips(image_path, output_path):
            logger.debug(f"Converted {image_path.name} using sips")
            return True

        # Fallback to Pillow
        if self.image_to_pdf_pillow(image_path, output_path):
            logger.debug(f"Converted {image_path.name} using Pillow")
            return True

        logger.error(f"Failed to convert {image_path}")
        return False

    def get_image_bytes(self, image_path: Path) -> tuple[bytes, str]:
        """Load image and return bytes with media type.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (image_bytes, media_type).
        """
        suffix = image_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return image_bytes, media_type
