"""Streamlit web interface for Receipt Analyzer."""

import io
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import streamlit as st

from analyze_receipts import discover_centers, process_center
from boundary_detector import BoundaryDetector
from claude_client import ClaudeClient, DEFAULT_CATEGORIZATION_RULES
from converter import ImageConverter
from excel_generator import ExcelGenerator
from models import Category, ProcessingResult
from pdf_processor import PDFProcessor


# Page config
st.set_page_config(
    page_title="Receipt Analyzer",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for senior-friendly UI
st.markdown("""
<style>
    .stButton > button {
        font-size: 18px;
        padding: 15px 30px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stTextArea > div > div > textarea {
        font-size: 14px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .stat-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .center-breakdown {
        background-color: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_svg_workflow():
    """Load and return the SVG workflow diagram."""
    svg_path = Path(__file__).parent / "assets" / "workflow.svg"
    if svg_path.exists():
        return svg_path.read_text()
    return None


def find_reimbursements_folder(extract_path: Path) -> Path:
    """Find the reimbursements folder in extracted ZIP.

    Args:
        extract_path: Path where ZIP was extracted.

    Returns:
        Path to the reimbursements folder.
    """
    # Check if extract_path itself contains center folders
    subdirs = [d for d in extract_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Look for a folder named 'reimbursements' or similar
    for subdir in subdirs:
        if 'reimbursement' in subdir.name.lower():
            return subdir

    # If there's exactly one folder, check inside it
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_subdirs = [d for d in inner.iterdir() if d.is_dir() and not d.name.startswith('.')]
        # Check if the inner folder contains center folders with receipt files
        for inner_subdir in inner_subdirs:
            if 'reimbursement' in inner_subdir.name.lower():
                return inner_subdir
        # Or maybe the single folder IS the reimbursements folder containing centers
        if inner_subdirs:
            return inner

    # Default: assume extract_path contains the center folders directly
    return extract_path


def create_output_zip(output_dir: Path) -> bytes:
    """Create a ZIP file of all output files.

    Args:
        output_dir: Directory containing output files.

    Returns:
        ZIP file as bytes.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                arcname = file_path.relative_to(output_dir)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def aggregate_statistics(results: list[ProcessingResult]) -> dict:
    """Aggregate statistics from all processing results.

    Args:
        results: List of ProcessingResult objects.

    Returns:
        Dictionary with aggregated statistics.
    """
    stats = {
        'centers_processed': len(results),
        'total_receipts': 0,
        'snack_receipts': 0,
        'supply_receipts': 0,
        'unknown_receipts': 0,
        'needs_review': 0,
        'total_amount': 0.0,
        'center_details': [],
    }

    for result in results:
        center_stats = {
            'name': result.center_name,
            'receipts': len(result.receipts),
            'amount': sum(r.total for r in result.receipts),
            'needs_review': sum(1 for r in result.receipts if r.needs_review),
            'errors': len(result.errors),
        }
        stats['center_details'].append(center_stats)

        stats['total_receipts'] += len(result.receipts)
        stats['total_amount'] += center_stats['amount']
        stats['needs_review'] += center_stats['needs_review']

        for receipt in result.receipts:
            if receipt.overall_category == Category.SNACK:
                stats['snack_receipts'] += 1
            elif receipt.overall_category == Category.SUPPLY:
                stats['supply_receipts'] += 1
            else:
                stats['unknown_receipts'] += 1

    return stats


def process_uploaded_zip(
    uploaded_file,
    api_key: str,
    custom_rules: str,
    progress_bar,
    status_text,
) -> tuple[bytes, dict, float]:
    """Process uploaded ZIP file and return results.

    Args:
        uploaded_file: Streamlit uploaded file.
        api_key: Claude API key.
        custom_rules: Custom categorization rules.
        progress_bar: Streamlit progress bar element.
        status_text: Streamlit text element for status.

    Returns:
        Tuple of (zip_bytes, statistics, processing_time).
    """
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract ZIP
        status_text.text("Extracting ZIP file...")
        progress_bar.progress(0.05)
        extract_path = temp_path / "input"
        extract_path.mkdir()

        with zipfile.ZipFile(uploaded_file, 'r') as zf:
            zf.extractall(extract_path)

        # Find reimbursements folder
        reimbursements_path = find_reimbursements_folder(extract_path)

        # Initialize clients
        status_text.text("Initializing AI analyzer...")
        progress_bar.progress(0.1)

        claude_client = ClaudeClient(
            api_key=api_key,
            custom_categorization_rules=custom_rules if custom_rules != DEFAULT_CATEGORIZATION_RULES else None,
        )
        boundary_detector = BoundaryDetector(api_key=api_key)
        pdf_processor = PDFProcessor()
        converter = ImageConverter()
        excel_generator = ExcelGenerator()

        # Discover centers
        centers = discover_centers(reimbursements_path)

        if not centers:
            raise ValueError("No center folders found in the uploaded ZIP. Please check the folder structure.")

        # Process each center
        output_path = temp_path / "output"
        output_path.mkdir()
        results = []

        total_centers = len(centers)
        for i, (center_name, files) in enumerate(centers.items()):
            progress = 0.1 + (0.8 * (i / total_centers))
            status_text.text(f"Processing {center_name}... ({i + 1}/{total_centers})")
            progress_bar.progress(progress)

            try:
                result = process_center(
                    center_name=center_name,
                    files=files,
                    output_dir=output_path,
                    claude_client=claude_client,
                    boundary_detector=boundary_detector,
                    pdf_processor=pdf_processor,
                    converter=converter,
                    excel_generator=excel_generator,
                )
                results.append(result)
            except Exception as e:
                # Create a result with the error
                error_result = ProcessingResult(center_name=center_name)
                error_result.errors.append(str(e))
                results.append(error_result)

        # Create output ZIP
        status_text.text("Creating download package...")
        progress_bar.progress(0.95)

        zip_bytes = create_output_zip(output_path)

        # Calculate statistics
        stats = aggregate_statistics(results)
        processing_time = time.time() - start_time

        progress_bar.progress(1.0)
        status_text.text("Complete!")

        return zip_bytes, stats, processing_time


def main():
    """Main Streamlit app."""

    # Header
    st.title("🧾 Receipt Analyzer")
    st.markdown("*Automatically categorize and organize childcare reimbursement receipts*")

    st.divider()

    # Workflow diagram
    svg_content = load_svg_workflow()
    if svg_content:
        st.markdown("### How it works")
        st.markdown(svg_content, unsafe_allow_html=True)
        st.divider()

    # Step 1: API Key
    st.markdown("### Step 1: Enter your Claude API Key")
    st.caption("Get your API key from [console.anthropic.com](https://console.anthropic.com)")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        label_visibility="collapsed",
        help="Your Claude API key is used to analyze receipts. It's never stored.",
    )

    # Step 2: File Upload
    st.markdown("### Step 2: Upload your receipts ZIP file")
    st.caption("ZIP should contain: `reimbursements/center_name/receipt_files`")

    uploaded_file = st.file_uploader(
        "Upload ZIP",
        type=["zip"],
        label_visibility="collapsed",
        help="Upload a ZIP file containing your receipt folders organized by center.",
    )

    # Advanced Options (collapsed)
    with st.expander("⚙️ Advanced Options"):
        st.markdown("**Categorization Rules**")
        st.caption("Edit these rules to customize how items are categorized:")

        custom_rules = st.text_area(
            "Rules",
            value=DEFAULT_CATEGORIZATION_RULES,
            height=250,
            label_visibility="collapsed",
        )

    st.divider()

    # Process button
    can_process = api_key and uploaded_file

    if st.button(
        "🚀 Process Receipts",
        disabled=not can_process,
        type="primary",
        use_container_width=True,
    ):
        if not api_key:
            st.error("Please enter your Claude API key.")
            return
        if not uploaded_file:
            st.error("Please upload a ZIP file.")
            return

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            zip_bytes, stats, processing_time = process_uploaded_zip(
                uploaded_file=uploaded_file,
                api_key=api_key,
                custom_rules=custom_rules,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            # Clear progress
            status_text.empty()
            progress_bar.empty()

            # Success message
            st.success("✅ Processing Complete!")

            # Time saved calculation (estimate 3 min per receipt manually)
            manual_time_minutes = stats['total_receipts'] * 3
            time_saved_minutes = manual_time_minutes - (processing_time / 60)
            time_saved_hours = time_saved_minutes / 60

            # Summary statistics
            st.markdown("### 📊 Summary")

            # Big time-saved message
            if time_saved_hours >= 1:
                st.markdown(
                    f"""<div class="success-box">
                    <h2 style="color: #155724; margin: 0;">🎉 You just saved approximately {time_saved_hours:.1f} hours!</h2>
                    <p style="margin: 10px 0 0 0; color: #155724;">
                    That's {stats['total_receipts']} receipts processed in just {processing_time:.0f} seconds.
                    </p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="success-box">
                    <h2 style="color: #155724; margin: 0;">🎉 You just saved approximately {time_saved_minutes:.0f} minutes!</h2>
                    <p style="margin: 10px 0 0 0; color: #155724;">
                    That's {stats['total_receipts']} receipts processed in just {processing_time:.0f} seconds.
                    </p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Stats in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Centers", stats['centers_processed'])
            with col2:
                st.metric("Total Receipts", stats['total_receipts'])
            with col3:
                st.metric("Snack Receipts", stats['snack_receipts'])
            with col4:
                st.metric("Supply Receipts", stats['supply_receipts'])

            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric("Needs Review", stats['needs_review'])
            with col6:
                st.metric("Total Amount", f"${stats['total_amount']:,.2f}")
            with col7:
                st.metric("Manual Time Est.", f"{manual_time_minutes} min")
            with col8:
                st.metric("AI Time", f"{processing_time:.0f} sec")

            # Download button
            st.divider()
            st.download_button(
                label="📥 Download Results (ZIP)",
                data=zip_bytes,
                file_name="receipt_analysis_results.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

            # Center breakdown
            if stats['center_details']:
                st.markdown("### 📋 Breakdown by Center")
                for center in stats['center_details']:
                    review_text = ""
                    if center['needs_review'] > 0:
                        review_text = f" - ⚠️ {center['needs_review']} need review"

                    error_text = ""
                    if center['errors'] > 0:
                        error_text = f" - ❌ {center['errors']} errors"

                    st.markdown(
                        f"""<div class="center-breakdown">
                        <strong>{center['name']}</strong>: {center['receipts']} receipts (${center['amount']:,.2f}){review_text}{error_text}
                        </div>""",
                        unsafe_allow_html=True,
                    )

        except zipfile.BadZipFile:
            st.error("❌ The uploaded file is not a valid ZIP file. Please try again.")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.caption("Please check your API key and try again. If the problem persists, check your receipt files.")

    # Help text
    if not can_process:
        st.info("👆 Enter your API key and upload a ZIP file to get started.")


if __name__ == "__main__":
    main()
