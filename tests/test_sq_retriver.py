import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.synthetic_questions.SQRetriver import SQRetriver

CSV_PATH = "evaluations/calibration/ques_creation/SyntheticQuestionKG.csv"


def test_init_resolves_default_calibration_location():
    retriever = SQRetriver()

    assert retriever.file_path.name == "SyntheticQuestionKG.csv"
    assert "evaluations/calibration/ques_creation" in str(retriever.file_path)


def test_category_specific_filters_are_inferred_from_sparse_columns():
    retriever = SQRetriver(CSV_PATH)

    assert retriever.get_category_specific_filters("path-level") == (
        "start_node",
        "end_node",
    )
    assert retriever.get_category_specific_filters("object-level|from-prop") == (
        "focal_relation",
        "focal_node",
    )
    assert retriever.get_category_specific_filters("object-level|from-object") == (
        "focal_relation",
        "focal_node",
    )


def test_retrieve_path_level_filters_by_path_nodes():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_path_level(
        start_node="prov:Association",
        end_node="provone:Program",
    )

    assert result
    assert {row["category"] for row in result} == {"path-level"}
    assert {row["start_node"] for row in result} == {"prov:Association"}
    assert {row["end_node"] for row in result} == {"provone:Program"}


def test_retrieve_path_level_filters_by_start_node():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_path_level(start_node="workflow:Generative_Task")

    assert result
    assert {row["category"] for row in result} == {"path-level"}
    assert {row["start_node"] for row in result} == {"workflow:Generative_Task"}


def test_retrieve_path_level_filters_by_end_node():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_path_level(end_node="provone:Port")

    assert result
    assert {row["category"] for row in result} == {"path-level"}
    assert {row["end_node"] for row in result} == {"provone:Port"}


def test_retrieve_object_level_from_object_filters_by_focal_node():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_object_level_from_object(focal_node="provone:Channel")

    assert result
    assert {row["category"] for row in result} == {"object-level|from-object"}
    assert {row["focal_node"] for row in result} == {"provone:Channel"}


def test_retrieve_object_level_from_object_filters_by_focal_relation():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_object_level_from_object(focal_relation="dc:description")

    assert result
    assert {row["category"] for row in result} == {"object-level|from-object"}
    assert {row["focal_relation"] for row in result} == {"dc:description"}


def test_retrieve_object_level_from_prop_filters_by_focal_node():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_object_level_from_prop(focal_node="provone:Channel")

    assert result
    assert {row["category"] for row in result} == {"object-level|from-prop"}
    assert {row["focal_node"] for row in result} == {"provone:Channel"}


def test_retrieve_object_level_from_prop_filters_by_focal_relation():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve_object_level_from_prop(focal_relation="dc:description")

    assert result
    assert {row["category"] for row in result} == {"object-level|from-prop"}
    assert {row["focal_relation"] for row in result} == {"dc:description"}


def test_retrieve_uses_category_aliases():
    retriever = SQRetriver(CSV_PATH)

    result = retriever.retrieve("from_prop", focal_node="provone:Channel")

    assert result
    assert {row["category"] for row in result} == {"object-level|from-prop"}
