"""KG adapter for running the vendored HyperGraphRAG baseline."""

from .kg_adapter import (
    HyperGraphKG,
    build_hypergraph_kg,
    write_hypergraph_kg,
)

__all__ = [
    "HyperGraphKG",
    "build_hypergraph_kg",
    "write_hypergraph_kg",
]

