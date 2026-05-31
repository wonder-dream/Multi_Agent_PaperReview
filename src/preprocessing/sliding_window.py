from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    index: int = 0


def chunk_text(
    text: str,
    window_size: int = 512,
    overlap: int = 128,
    tokenizer_name: Optional[str] = None,
) -> List[Chunk]:
    """Split long text into overlapping chunks.

    If tokenizer_name is provided, chunking respects token boundaries
    using the HuggingFace tokenizer. Otherwise, simple character-level
    chunking is used (window_size is in characters).

    Args:
        text: Input text to chunk.
        window_size: Size of each chunk (in tokens if tokenizer, else characters).
        overlap: Number of overlapping units between consecutive chunks.
        tokenizer_name: Optional HuggingFace tokenizer name for token-level chunking.

    Returns:
        List of Chunk objects with text, start/end character offsets, and index.
    """
    if not text or not text.strip():
        return []

    if tokenizer_name is not None:
        return _token_level_chunks(text, window_size, overlap, tokenizer_name)
    return _char_level_chunks(text, window_size, overlap)


def _char_level_chunks(text: str, window_size: int, overlap: int) -> List[Chunk]:
    """Character-level sliding window."""
    chunks = []
    start = 0
    text_len = len(text)
    idx = 0

    while start < text_len:
        end = min(start + window_size, text_len)

        # Try to break at a sentence boundary (period followed by space or newline)
        if end < text_len:
            lookback_start = max(start, end - overlap)
            for sep in [". ", ".\n", "? ", "!\n", "\n\n"]:
                pos = text.rfind(sep, lookback_start, end)
                if pos > lookback_start:
                    end = pos + 1  # include the period
                    break

        chunk_text_value = text[start:end].strip()
        if chunk_text_value:
            chunks.append(Chunk(text=chunk_text_value, start=start, end=end, index=idx))
            idx += 1

        if end >= text_len:
            break
        start = end - overlap

    return chunks


def _token_level_chunks(
    text: str, window_size: int, overlap: int, tokenizer_name: str
) -> List[Chunk]:
    """Token-level sliding window using a HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []
    start = 0
    idx = 0
    total = len(input_ids)

    while start < total:
        end = min(start + window_size, total)

        if end < total and end > start:
            token_text = tokenizer.decode(input_ids[end - overlap : end])
            if ". " in token_text or ".\n" in token_text:
                for sep_token in [".", "?\n", "!\n"]:
                    sep_ids = tokenizer.encode(sep_token, add_special_tokens=False)
                    if sep_ids:
                        search_start = max(start, end - overlap)
                        for i in range(end - 1, search_start - 1, -1):
                            if input_ids[i] == sep_ids[0]:
                                end = i + 1
                                break

        char_start = offsets[start][0] if start < len(offsets) else 0
        char_end = offsets[end - 1][1] if end > 0 and end - 1 < len(offsets) else len(text)

        chunk_text_value = text[char_start:char_end].strip()
        if chunk_text_value:
            chunks.append(Chunk(text=chunk_text_value, start=char_start, end=char_end, index=idx))
            idx += 1

        if end >= total:
            break
        start = end - overlap

    return chunks
