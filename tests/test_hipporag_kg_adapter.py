import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.HippoRAGKG import build_kg_documents, write_hipporag_openie


def test_build_kg_documents_and_injected_openie(tmp_path):
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

    documents = build_kg_documents(kg_path, ontology_path)

    run_doc = next(document for document in documents if document.id == "ex:run1")
    assert "Run 1" in run_doc.title
    assert "ex:Execution" in run_doc.types
    assert ["Run 1 (ex:run1)", "used input", "Input 1 (ex:input1)"] in run_doc.extracted_triples
    assert ["Run 1 (ex:run1)", "status", "completed"] in run_doc.extracted_triples

    openie_path = write_hipporag_openie(documents, tmp_path / "openie.json")
    payload = json.loads(openie_path.read_text(encoding="utf-8"))
    assert len(payload["docs"]) == len(documents)
    assert payload["docs"][0]["idx"].startswith("chunk-")
    assert {"idx", "passage", "extracted_entities", "extracted_triples"} <= set(
        payload["docs"][0]
    )
