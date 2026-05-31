import os
import subprocess
import tempfile
import shutil


def parse_pdf(path: str) -> str:
    """Extract text from a PDF file using MinerU CLI, with PyMuPDF fallback.

    MinerU produces high-quality markdown including tables and formulas.
    Falls back to PyMuPDF if MinerU is unavailable or fails.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted markdown text (MinerU) or plain text (PyMuPDF fallback).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    result = _try_mineru(path)
    if result is not None:
        return result
    return _fallback_pymupdf(path)


def _try_mineru(path: str) -> str | None:
    """Try MinerU flash-extract, then extract with token."""
    for mode in ("flash-extract", "extract"):
        try:
            out_dir = tempfile.mkdtemp(prefix="mineru_")
            cmd = ["mineru-open-api", mode, os.path.abspath(path), "-o", out_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # MinerU outputs to out_dir/<filename>/<filename>.md
                base = os.path.splitext(os.path.basename(path))[0]
                md_path = os.path.join(out_dir, base, f"{base}.md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    shutil.rmtree(out_dir, ignore_errors=True)
                    if text.strip():
                        return text
            shutil.rmtree(out_dir, ignore_errors=True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _fallback_pymupdf(path: str) -> str:
    """Fallback: extract plain text using PyMuPDF."""
    import fitz

    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text:
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)
