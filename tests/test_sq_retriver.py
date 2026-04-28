import sys
from pathlib import Path

import pandas as pd
import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.synthetic_questions.SQRetriver import SQRetriver


@pytest.fixture
def synthetic_questions_csv(tmp_path):
    csv_path = tmp_path / "SyntheticQuestionKG.csv"
    rows = [
        {
            "program_id": "from_prop_artist",
            "solves": "Find programs from artist relation",
            "category": "object-level|from-prop",
            "focal_relation": "rel:artist",
            "focal_node": "class:Song",
            "start_node": "",
            "end_node": "",
        },
        {
            "program_id": "from_prop_album",
            "solves": "Find programs from album relation",
            "category": "object-level|from-prop",
            "focal_relation": "rel:album",
            "focal_node": "class:Album",
            "start_node": "",
            "end_node": "",
        },
        {
            "program_id": "from_object_artist",
            "solves": "Find objects from artist relation",
            "category": "object-level|from-object",
            "focal_relation": "rel:artist",
            "focal_node": "class:Song",
            "start_node": "",
            "end_node": "",
        },
        {
            "program_id": "path_song_album",
            "solves": "Find song to album path",
            "category": "path-level",
            "focal_relation": "",
            "focal_node": "",
            "start_node": "class:Song",
            "end_node": "class:Album",
        },
        {
            "program_id": "path_artist_song",
            "solves": "Find artist to song path",
            "category": "path-level",
            "focal_relation": "",
            "focal_node": "",
            "start_node": "class:Artist",
            "end_node": "class:Song",
        },
        {
            "program_id": "path_artist_album",
            "solves": "Find artist to album path",
            "category": "path-level",
            "focal_relation": "",
            "focal_node": "",
            "start_node": "class:Artist",
            "end_node": "class:Album",
        },
        {
            "program_id": "explore_object_of_class",
            "solves": "List objects of a class",
            "category": "seed",
            "focal_relation": "",
            "focal_node": "",
            "start_node": "",
            "end_node": "",
        },
        {
            "program_id": "explore_attr_of_object",
            "solves": "List data attributes of an object",
            "category": "seed",
            "focal_relation": "",
            "focal_node": "",
            "start_node": "",
            "end_node": "",
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def record_ids(records):
    return [record["program_id"] for record in records]


def test_init_loads_rows_from_location(synthetic_questions_csv):
    retriever = SQRetriver(str(synthetic_questions_csv))

    assert len(retriever.rows) == 8
    assert list(retriever.rows["program_id"])[:2] == [
        "from_prop_artist",
        "from_prop_album",
    ]


def test_get_programs_from_prop_filters_by_relation_and_category(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_programs_from_prop("rel:artist")

    assert record_ids(records) == ["from_prop_artist"]
    assert [record["solves"] for record in records] == [
        "Find programs from artist relation"
    ]


def test_get_programs_from_prop_can_filter_by_class_candidates(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_programs_from_prop(
        "rel:artist",
        class_candidates=["class:Song"],
    )

    assert record_ids(records) == ["from_prop_artist"]


def test_get_programs_from_prop_returns_only_program_id_and_solves_fields(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_programs_from_prop("rel:album")

    assert records.dtype.names == ("index", "program_id", "solves")


def test_get_objects_of_a_class_returns_class_explorer_seed_row(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    record = retriever.get_objects_of_a_class("class:Song")

    assert record["program_id"] == "explore_object_of_class"
    assert record["solves"] == "List objects of a class"


def test_get_data_of_object_returns_object_attribute_seed_row(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    record = retriever.get_data_of_object("entity:song-1")

    assert record["program_id"] == "explore_attr_of_object"
    assert record["solves"] == "List data attributes of an object"


def test_get_path_questions_returns_all_path_level_questions(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_path_questions()

    assert record_ids(records) == [
        "path_song_album",
        "path_artist_song",
        "path_artist_album",
    ]


def test_get_path_questions_filters_by_start_nodes(synthetic_questions_csv):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_path_questions(start_nodes=["class:Artist"])

    assert record_ids(records) == ["path_artist_song", "path_artist_album"]


def test_get_path_questions_filters_by_end_nodes(synthetic_questions_csv):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_path_questions(end_nodes=["class:Album"])

    assert record_ids(records) == ["path_song_album", "path_artist_album"]


def test_get_path_questions_filters_by_start_and_end_nodes(
    synthetic_questions_csv,
):
    retriever = SQRetriver(str(synthetic_questions_csv))

    records = retriever.get_path_questions(
        start_nodes=["class:Song"],
        end_nodes=["class:Album"],
    )

    assert record_ids(records) == ["path_song_album"]


def test_get_program_by_id_returns_matching_row(synthetic_questions_csv):
    retriever = SQRetriver(str(synthetic_questions_csv))

    record = retriever.get_program_by_id("from_prop_album")

    assert record["program_id"] == "from_prop_album"
    assert record["category"] == "object-level|from-prop"
