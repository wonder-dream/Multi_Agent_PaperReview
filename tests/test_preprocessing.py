"""Tests for preprocessing module: PDF parser + sliding window chunker."""
import os
import tempfile
import pytest

# We'll import the real module once it exists
# from src.preprocessing.pdf_parser import parse_pdf
# from src.preprocessing.sliding_window import chunk_text, Chunk


class TestParsePdf:
    """Tests for parse_pdf(path) -> str."""

    def test_parses_text_from_pdf(self):
        """A valid PDF should return non-empty text."""
        from src.preprocessing.pdf_parser import parse_pdf

        # Create a minimal text-based PDF
        pdf_path = _create_minimal_pdf("Hello World. This is a test paper.")

        text = parse_pdf(pdf_path)
        assert len(text) > 0
        assert "Hello World" in text

    def test_raises_on_nonexistent_file(self):
        """Should raise FileNotFoundError for missing files."""
        from src.preprocessing.pdf_parser import parse_pdf

        with pytest.raises(FileNotFoundError):
            parse_pdf("/tmp/does_not_exist_xyz.pdf")

    def test_returns_empty_string_for_empty_pdf(self):
        """Empty PDF should return empty string, not crash."""
        from src.preprocessing.pdf_parser import parse_pdf

        pdf_path = _create_minimal_pdf("")
        text = parse_pdf(pdf_path)
        assert text == ""


class TestChunkText:
    """Tests for chunk_text(text, tokenizer, window, overlap) -> List[Chunk]."""

    def test_chunks_short_text_into_single_chunk(self):
        """Text shorter than window_size should produce one chunk."""
        from src.preprocessing.sliding_window import chunk_text

        text = "Short text."
        chunks = chunk_text(text, window_size=512, overlap=128)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start == 0

    def test_chunks_long_text_into_multiple(self):
        """Text longer than window_size should produce multiple chunks."""
        from src.preprocessing.sliding_window import chunk_text

        # Use a tiny window to force chunking of a known text
        text = "Token1 Token2 Token3 Token4 Token5 Token6 Token7 Token8 Token9 Token10"
        chunks = chunk_text(text, window_size=3, overlap=0)

        assert len(chunks) > 1
        # All chunks together should cover the complete text (except overlap)
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunks_have_correct_positions(self):
        """Each chunk should record its start/end position in the original text."""
        from src.preprocessing.sliding_window import chunk_text

        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH"
        chunks = chunk_text(text, window_size=3, overlap=0)

        for i, chunk in enumerate(chunks):
            assert chunk.start <= len(text)
            assert chunk.end <= len(text)
            assert chunk.start < chunk.end
            # Text extracted at [start:end] should match
            assert text[chunk.start:chunk.end].strip() == chunk.text.strip() or \
                   text[chunk.start:chunk.end] in chunk.text or \
                   chunk.text in text[chunk.start:chunk.end]

    def test_overlap_reduces_total_chunks(self):
        """Larger overlap should produce fewer total chunks."""
        from src.preprocessing.sliding_window import chunk_text

        text = " ".join([f"Token{i}" for i in range(50)])
        chunks_no_overlap = chunk_text(text, window_size=5, overlap=0)
        chunks_with_overlap = chunk_text(text, window_size=5, overlap=2)

        assert len(chunks_with_overlap) > len(chunks_no_overlap)

    def test_empty_text_returns_no_chunks(self):
        """Empty input should return empty list."""
        from src.preprocessing.sliding_window import chunk_text

        chunks = chunk_text("", window_size=512, overlap=128)
        assert chunks == []


# --- helpers ---

def _create_minimal_pdf(text: str) -> str:
    """Create a temporary PDF file with the given text content.
    Returns the file path. Caller is responsible for cleanup.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Try pypdf as fallback
        try:
            from pypdf import PdfWriter
        except ImportError:
            pytest.skip("Neither PyMuPDF nor pypdf is installed")

    # Use pypdf to create a minimal valid PDF
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawString(100, 750, text)
    c.save()
    buf.seek(0)

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(buf.read())
    return path
