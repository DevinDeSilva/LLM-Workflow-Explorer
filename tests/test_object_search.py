import sys
from pathlib import Path

import pandas as pd

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.config.object_search import ObjectSearchConfig
from src.explainer.object_search import ObjectSearch


class FakeGraphManager:
    def __init__(self):
        self.query_calls = []
        self.objects = [
            {"object": "http://example.org/Object1"},
            {"object": "http://example.org/Object2"},
            {"object": "http://example.org/Object1"},
        ]
        self.properties = {
            "http://example.org/Object1": [
                {
                    "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "value": "http://example.org/Widget",
                },
                {
                    "predicate": "http://example.org/hasLabel",
                    "value": "Primary widget",
                },
                {
                    "predicate": "http://example.org/connectedTo",
                    "value": "http://example.org/Object2",
                },
            ],
            "http://example.org/Object2": [
                {
                    "predicate": "http://example.org/hasLabel",
                    "value": "Secondary widget",
                },
            ],
        }

    def query(self, sparql_query: str):
        self.query_calls.append(sparql_query)

        if "?object" in sparql_query or "ALL_OBJECTS_QUERY" in sparql_query:
            return pd.DataFrame(self.objects)

        object_uri = sparql_query.split("<", 1)[1].split(">", 1)[0]
        return pd.DataFrame(self.properties.get(object_uri, []))

    def reverse_curie(self, value: str) -> str:
        mapping = {
            "http://example.org/Object1": "ex:Object1",
            "http://example.org/Object2": "ex:Object2",
            "http://example.org/Widget": "ex:Widget",
            "http://example.org/hasLabel": "ex:hasLabel",
            "http://example.org/connectedTo": "ex:connectedTo",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "rdf:type",
        }
        return mapping.get(value, value)


class FakeEmbeddingsModel:
    def __init__(self):
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, texts):
        self.document_inputs.append(list(texts))
        return [[float(index), float(index) + 0.5] for index, _ in enumerate(texts, start=1)]

    def embed_query(self, text):
        self.query_inputs.append(text)
        return [0.1, 0.2]


class FakeVectorDB:
    def __init__(self):
        self.build_db_calls = []
        self.insert_calls = []
        self.search_calls = []
        self.bm25_search_calls = []

    def build_db(self, overwrite: bool = False):
        self.build_db_calls.append(overwrite)
        return "calibration"

    def insert(self, records=None, build_if_missing: bool = True, **kwargs):
        self.insert_calls.append(
            {
                "records": records,
                "build_if_missing": build_if_missing,
                "kwargs": kwargs,
            }
        )
        return {"insert_count": len(records or [])}

    def search(self, query_vector, limit: int = 10, **kwargs):
        self.search_calls.append(
            {
                "query_vector": query_vector,
                "limit": limit,
                "kwargs": kwargs,
            }
        )
        return [{"object_name": "ex:Object1", "score": 0.99}]

    def bm25_search(self, query_text: str, limit: int = 10, **kwargs):
        self.bm25_search_calls.append(
            {
                "query_text": query_text,
                "limit": limit,
                "kwargs": kwargs,
            }
        )
        return [{"object_name": "ex:Object2", "score": 0.85}]


def test_build_index_records_creates_descriptions_and_vectors():
    graph_manager = FakeGraphManager()
    embeddings_model = FakeEmbeddingsModel()
    vector_db = FakeVectorDB()

    object_search = ObjectSearch(
        graph_manager=graph_manager,
        embeddings_model=embeddings_model,
        vector_db=vector_db,
    )

    records = object_search.build_index_records()

    assert [record["object_name"] for record in records] == [
        "ex:Object1",
        "ex:Object2",
    ]
    assert records[0]["metadata"] == {}
    assert records[0]["object_vector"] == [1.0, 1.5]
    assert records[0]["object_description"] == (
        "Object: ex:Object1. "
        "Types: ex:Widget. "
        "Properties: ex:connectedTo: ex:Object2; ex:hasLabel: Primary widget."
    )
    assert records[1]["object_description"] == (
        "Object: ex:Object2. "
        "Properties: ex:hasLabel: Secondary widget."
    )
    assert embeddings_model.document_inputs == [
        [
            "Object: ex:Object1. Types: ex:Widget. Properties: ex:connectedTo: ex:Object2; ex:hasLabel: Primary widget.",
            "Object: ex:Object2. Properties: ex:hasLabel: Secondary widget.",
        ]
    ]


def test_index_objects_overwrite_rebuilds_collection_before_insert():
    graph_manager = FakeGraphManager()
    embeddings_model = FakeEmbeddingsModel()
    vector_db = FakeVectorDB()

    object_search = ObjectSearch(
        graph_manager=graph_manager,
        embeddings_model=embeddings_model,
        vector_db=vector_db,
    )

    result = object_search.index_objects(overwrite=True)

    assert result == {"insert_count": 2}
    assert vector_db.build_db_calls == [True]
    assert vector_db.insert_calls[0]["build_if_missing"] is False
    assert len(vector_db.insert_calls[0]["records"]) == 2


def test_search_returns_vector_and_bm25_results():
    graph_manager = FakeGraphManager()
    embeddings_model = FakeEmbeddingsModel()
    vector_db = FakeVectorDB()

    object_search = ObjectSearch(
        graph_manager=graph_manager,
        embeddings_model=embeddings_model,
        vector_db=vector_db,
    )

    results = object_search.search("primary widget", limit=5)

    assert results == {
        "vector_results": [{"object_name": "ex:Object1", "score": 0.99}],
        "bm25_results": [{"object_name": "ex:Object2", "score": 0.85}],
    }
    assert embeddings_model.query_inputs == ["primary widget"]
    assert vector_db.search_calls == [
        {
            "query_vector": [0.1, 0.2],
            "limit": 5,
            "kwargs": {},
        }
    ]
    assert vector_db.bm25_search_calls == [
        {
            "query_text": "primary widget",
            "limit": 5,
            "kwargs": {},
        }
    ]


def test_search_uses_default_limit_from_config():
    graph_manager = FakeGraphManager()
    embeddings_model = FakeEmbeddingsModel()
    vector_db = FakeVectorDB()
    config = ObjectSearchConfig(search_limit=3)

    object_search = ObjectSearch(
        graph_manager=graph_manager,
        config=config,
        embeddings_model=embeddings_model,
        vector_db=vector_db,
    )

    results = object_search.search("primary widget")

    assert results == {
        "vector_results": [{"object_name": "ex:Object1", "score": 0.99}],
        "bm25_results": [{"object_name": "ex:Object2", "score": 0.85}],
    }
    assert vector_db.search_calls[0]["limit"] == 3
    assert vector_db.bm25_search_calls[0]["limit"] == 3


def test_queries_are_read_from_config():
    graph_manager = FakeGraphManager()
    embeddings_model = FakeEmbeddingsModel()
    vector_db = FakeVectorDB()
    config = ObjectSearchConfig(
        all_objects_query="ALL_OBJECTS_QUERY",
        object_properties_query="OBJECT_PROPERTIES_QUERY <{object_uri}>",
    )

    object_search = ObjectSearch(
        graph_manager=graph_manager,
        config=config,
        embeddings_model=embeddings_model,
        vector_db=vector_db,
    )

    records = object_search.build_index_records()

    assert len(records) == 2
    assert graph_manager.query_calls[0] == "ALL_OBJECTS_QUERY"
    assert graph_manager.query_calls[1] == "OBJECT_PROPERTIES_QUERY <http://example.org/Object1>"
    assert graph_manager.query_calls[2] == "OBJECT_PROPERTIES_QUERY <http://example.org/Object2>"
