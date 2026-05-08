from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_HyperGraphRAG import (
    _parse_related_entities,
    _prediction_from_context,
)


def test_parse_related_entities_handles_hypergraphrag_formats() -> None:
    assert _parse_related_entities("['Biomni:a', 'Biomni:b']") == [
        "Biomni:a",
        "Biomni:b",
    ]
    assert _parse_related_entities("Biomni:a<SEP>Biomni:b") == [
        "Biomni:a",
        "Biomni:b",
    ]
    assert _parse_related_entities("Biomni:a | Biomni:b; Biomni:c") == [
        "Biomni:a",
        "Biomni:b",
        "Biomni:c",
    ]


def test_prediction_from_context_populates_retrieval_fields() -> None:
    context = """
-----Entities-----
```csv
id,entity,type,description
0,Biomni:Data-id_1,workflow:Data,First data object
1,Biomni:Program-id_2,provone:Program,Program object
```
-----Relationships-----
```csv
id,hyperedge,related_entities
0,Biomni:edge_1,"['Biomni:Data-id_1', 'Biomni:Program-id_2']"
1,Biomni:edge_2,Biomni:Port-id_3<SEP>Biomni:Data-id_4
```
-----Sources-----
```csv
id,content
0,"Biomni:Data-id_1 [First data object]"
```
"""

    pred = _prediction_from_context(
        qinfo=SimpleNamespace(question="What was retrieved?", id="gt_test"),
        answer="A structured answer.",
        context=context,
        token_usage={"total_tokens": 0},
        time_taken=1.25,
    )

    assert [entity["id"] for entity in pred["relevant_entities"]] == [
        "Biomni:Data-id_1",
        "Biomni:Program-id_2",
    ]
    assert [row["object_id"] for row in pred["evidence"]] == [
        "Biomni:Data-id_1",
        "Biomni:Program-id_2",
        "Biomni:Port-id_3",
        "Biomni:Data-id_4",
    ]
    assert pred["retrieved_docs"] == [
        {"text": "Biomni:Data-id_1 [First data object]", "score": 0.0}
    ]
    assert pred["hypergraphrag_context_parse_status"] == "parsed"
    assert pred["hypergraphrag_context_entity_rows"] == 2
    assert pred["hypergraphrag_context_relationship_rows"] == 2
    assert pred["hypergraphrag_context_source_rows"] == 1


def test_prediction_from_context_marks_fail_response() -> None:
    pred = _prediction_from_context(
        qinfo=SimpleNamespace(question="What was retrieved?", id="gt_test"),
        answer="Sorry, I'm not able to provide an answer to that question.",
        context="Sorry, I'm not able to provide an answer to that question.",
        token_usage={"total_tokens": 0},
        time_taken=0.5,
    )

    assert pred["relevant_entities"] == []
    assert pred["evidence"] == []
    assert pred["retrieved_docs"] == []
    assert pred["hypergraphrag_context_parse_status"] == "fail_response"
