import csv
import json
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.utils.ontology_info_retriever import OntologyInfoRetriever


def test_format_schema_prompt_returns_compact_prompt_context(tmp_path):
    triples_path = tmp_path / "ontology.csv"
    schema_path = tmp_path / "schema.json"

    with open(triples_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["s", "p", "o"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "s": "workflow:Task",
                    "p": "rdfs:label",
                    "o": "Generative Task",
                },
                {
                    "s": "workflow:Task",
                    "p": "rdfs:comment",
                    "o": "Represents a user-facing task completed with generative AI support.",
                },
                {
                    "s": "workflow:Model",
                    "p": "rdfs:label",
                    "o": "Large Language Model",
                },
                {
                    "s": "workflow:Model",
                    "p": "http://purl.org/dc/terms/alternative",
                    "o": "LLM",
                },
                {
                    "s": "workflow:Model",
                    "p": "prov:definition",
                    "o": "A model used to generate or transform text.",
                },
                {
                    "s": "workflow:usesModel",
                    "p": "rdfs:label",
                    "o": "usesModel",
                },
                {
                    "s": "workflow:usesModel",
                    "p": "rdfs:comment",
                    "o": "Connects a task to the model selected to perform it while keeping the workflow explanation easy to follow.",
                },
            ]
        )

    schema = {
        "classes": ["workflow:Task", "workflow:Model"],
        "object_properties": {
            "workflow:usesModel": {
                "connections": [
                    {
                        "domain": "workflow:Task",
                        "range": "workflow:Model",
                    }
                ]
            }
        },
    }
    schema_path.write_text(json.dumps(schema))

    retriever = OntologyInfoRetriever(
        str(triples_path),
        schema_path=str(schema_path),
    )

    prompt = retriever.format_schema_prompt(description_limit=70)

    assert prompt.splitlines() == [
        "Workflow schema",
        "Classes:",
        "- workflow:Task",
        "- workflow:Model",
        "Object properties:",
        "- workflow:usesModel: workflow:Task -> workflow:Model",
    ]
