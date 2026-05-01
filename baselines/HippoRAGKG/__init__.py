"""KG adapter for running the vendored HippoRAG baseline."""

from .kg_adapter import (
    KGDocument,
    KGObjectDocumentBuilder,
    build_kg_documents,
    documents_by_content,
    write_hipporag_corpus,
    write_hipporag_openie,
)

__all__ = [
    "KGDocument",
    "KGObjectDocumentBuilder",
    "build_kg_documents",
    "documents_by_content",
    "write_hipporag_corpus",
    "write_hipporag_openie",
]

