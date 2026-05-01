"""
tests/test_chunking.py

Unit tests for the document ingestion pipeline.
These run in CI without needing Ollama or ChromaDB — pure logic tests.
"""

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def splitter():
    """Default splitter matching ingest.py config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )


@pytest.fixture
def short_doc():
    return Document(
        page_content="This is a short document. It has two sentences.",
        metadata={"source": "test.pdf", "page": 0},
    )


@pytest.fixture
def long_doc():
    # ~1500 chars — should produce multiple chunks
    paragraph = (
        "Climate change poses significant risks to agricultural systems across "
        "sub-Saharan Africa. Rising temperatures and shifting rainfall patterns "
        "are already affecting crop yields, water availability, and food security "
        "for millions of smallholder farmers. Adaptation strategies must be "
        "context-specific, drawing on both indigenous knowledge and modern "
        "agronomic research. "
    )
    return Document(
        page_content=paragraph * 6,
        metadata={"source": "climate_report.pdf", "page": 1},
    )


@pytest.fixture
def multi_doc(short_doc, long_doc):
    return [short_doc, long_doc]


# ---------------------------------------------------------------------------
# Chunk size tests
# ---------------------------------------------------------------------------

class TestChunkSize:
    def test_no_chunk_exceeds_max_size(self, splitter, long_doc):
        chunks = splitter.split_documents([long_doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= 800, (
                f"Chunk exceeded max size: {len(chunk.page_content)} chars\n"
                f"Content: {chunk.page_content[:80]}..."
            )

    def test_short_doc_is_not_split(self, splitter, short_doc):
        chunks = splitter.split_documents([short_doc])
        assert len(chunks) == 1, (
            f"Short doc should stay as one chunk, got {len(chunks)}"
        )

    def test_long_doc_produces_multiple_chunks(self, splitter, long_doc):
        chunks = splitter.split_documents([long_doc])
        assert len(chunks) > 1, (
            "Long doc (~1500 chars) should produce more than one chunk"
        )

    def test_no_empty_chunks(self, splitter, multi_doc):
        chunks = splitter.split_documents(multi_doc)
        for chunk in chunks:
            assert chunk.page_content.strip() != "", "Found empty chunk"


# ---------------------------------------------------------------------------
# Overlap tests
# ---------------------------------------------------------------------------

class TestChunkOverlap:
    def test_adjacent_chunks_share_content(self, splitter, long_doc):
        """Consecutive chunks should share at least some text (overlap > 0)."""
        chunks = splitter.split_documents([long_doc])
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        overlaps_found = 0
        for i in range(len(chunks) - 1):
            a = chunks[i].page_content
            b = chunks[i + 1].page_content
            # Check for any shared substring of 20+ chars
            for start in range(0, len(a) - 20, 10):
                snippet = a[start:start + 20]
                if snippet in b:
                    overlaps_found += 1
                    break

        assert overlaps_found > 0, (
            "Expected at least one pair of adjacent chunks to share content "
            "(chunk_overlap=150 should guarantee this)"
        )

    def test_overlap_does_not_duplicate_entire_chunk(self, splitter, long_doc):
        """Overlap should be partial — not a full copy of the previous chunk."""
        chunks = splitter.split_documents([long_doc])
        for i in range(1, len(chunks)):
            assert chunks[i].page_content != chunks[i - 1].page_content, (
                f"Chunk {i} is identical to chunk {i-1} — overlap may be misconfigured"
            )


# ---------------------------------------------------------------------------
# Metadata preservation tests
# ---------------------------------------------------------------------------

class TestMetadataPreservation:
    def test_source_metadata_preserved(self, splitter, long_doc):
        chunks = splitter.split_documents([long_doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "climate_report.pdf", (
                "Source metadata was lost during splitting"
            )

    def test_page_metadata_preserved(self, splitter, long_doc):
        chunks = splitter.split_documents([long_doc])
        for chunk in chunks:
            assert "page" in chunk.metadata, (
                "Page metadata was lost during splitting"
            )

    def test_metadata_preserved_across_multiple_docs(self, splitter, multi_doc):
        chunks = splitter.split_documents(multi_doc)
        sources = {chunk.metadata.get("source") for chunk in chunks}
        assert "test.pdf" in sources
        assert "climate_report.pdf" in sources


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_document_list(self, splitter):
        chunks = splitter.split_documents([])
        assert chunks == []

    def test_document_with_only_whitespace(self, splitter):
        doc = Document(page_content="   \n\n   \t   ", metadata={"source": "blank.pdf"})
        chunks = splitter.split_documents([doc])
        # Either no chunks or only whitespace chunks — neither should crash
        for chunk in chunks:
            # We just assert no exception was raised; content may be whitespace
            assert isinstance(chunk.page_content, str)

    def test_document_with_special_characters(self, splitter):
        doc = Document(
            page_content=(
                "Nairobi (population: 4.4M) recorded rainfall of 850mm in 2023. "
                "The NDVI index showed a 12% decrease (p<0.05) in arid zones. "
                "Key findings: CO₂ levels rose by 2.3 ppm; temperatures by +0.8°C."
            ) * 10,
            metadata={"source": "kenya_stats.pdf"},
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 800

    def test_splitter_config_chunk_size(self, splitter):
        """Sanity check: confirm splitter is configured as ingest.py expects."""
        assert splitter._chunk_size == 800
        assert splitter._chunk_overlap == 150

    def test_total_content_not_lost(self, splitter, long_doc):
        """
        All words in the original doc should appear in at least one chunk.
        (Overlap means total chars across chunks > original, but no word drops.)
        """
        chunks = splitter.split_documents([long_doc])
        all_chunk_text = " ".join(c.page_content for c in chunks)
        original_words = long_doc.page_content.split()

        missing = [w for w in original_words if w not in all_chunk_text]
        assert len(missing) == 0, (
            f"{len(missing)} words from the original document not found in any chunk"
        )