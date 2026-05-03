import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.HyperGraphRAGKG import build_hypergraph_kg


def test_build_hypergraph_kg_from_rdf_triples(tmp_path):
    kg_path = tmp_path / "kg.ttl"
    ontology_path = tmp_path / "ontology.ttl"
    kg_path.write_text(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

        ex:run1 a ex:Execution ;
          rdfs:label "Run 1" ;
          ex:used ex:input1 ;
          ex:status "completed" .

        ex:input1 rdfs:label "Input 1" .
        """,
        encoding="utf-8",
    )
    ontology_path.write_text(
        """
        @prefix ex: <http://example.org/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

        ex:used rdfs:label "used input" .
        ex:status rdfs:label "status" .
        """,
        encoding="utf-8",
    )

    custom_kg = build_hypergraph_kg(kg_path, ontology_path)
    payload = custom_kg.to_dict()

    assert payload["chunks"]
    assert any(entity["entity_name"] == '"RUN 1 (EX:RUN1)"' for entity in payload["entities"])
    assert any(entity["entity_name"] == '"INPUT 1 (EX:INPUT1)"' for entity in payload["entities"])
    assert any(entity["entity_type"] == "LITERAL" for entity in payload["entities"])
    assert any(
        edge["hyperedge_name"].startswith("<hyperedge>used input:")
        and '"RUN 1 (EX:RUN1)"' in edge["entity_names"]
        and '"INPUT 1 (EX:INPUT1)"' in edge["entity_names"]
        for edge in payload["hyperedges"]
    )

