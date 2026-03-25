from src.planning.dependancy_graph import DependancyGraphCreation


def test_normalize_plan_builds_edges_from_dependencies():
    raw_plan = {
        "answer_type": "count",
        "entities": ["Dataset", "Workflow"],
        "requirements": [
            {
                "id": "resolve_dataset",
                "label": "Resolve dataset",
                "description": "Find the referenced dataset.",
                "requirement_type": "entity_resolution",
                "depends_on": [],
            },
            {
                "id": "find_runs",
                "label": "Find related workflow runs",
                "description": "Retrieve workflow runs for the dataset.",
                "requirement_type": "graph_traversal",
                "depends_on": ["resolve_dataset"],
            },
        ],
        "edges": [],
    }

    normalized = DependancyGraphCreation._normalize_plan(
        raw_plan,
        query="How many workflow runs used Dataset X?",
    )

    assert normalized["answer_type"] == "count"
    assert normalized["execution_order"] == ["resolve_dataset", "find_runs"]
    assert normalized["edges"] == [
        {"source": "resolve_dataset", "target": "find_runs"}
    ]


def test_normalize_plan_falls_back_when_requirements_missing():
    normalized = DependancyGraphCreation._normalize_plan(
        {"answer_type": "entity"},
        query="Who generated the report for Experiment Alpha?",
    )

    assert len(normalized["requirements"]) == 2
    assert normalized["execution_order"] == [
        normalized["requirements"][0]["id"],
        "answer_projection",
    ]
    assert normalized["entities"] == []


def test_parse_plan_json_extracts_embedded_json():
    raw_output = """
    planner response:
    {
      "answer_type": "entity",
      "entities": ["Alpha"],
      "requirements": []
    }
    """

    parsed = DependancyGraphCreation._parse_plan_json(raw_output)

    assert parsed["answer_type"] == "entity"
    assert parsed["entities"] == ["Alpha"]
