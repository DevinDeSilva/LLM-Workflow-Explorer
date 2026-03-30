import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.config.vector_db.milvus import MilvusVectorDBConfig
import src.vector_db.milvus as milvus_module


class FakeDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    SPARSE_FLOAT_VECTOR = "SPARSE_FLOAT_VECTOR"
    JSON = "JSON"


class FakeFunctionType:
    BM25 = "BM25"


class FakeFunction:
    def __init__(
        self,
        name: str,
        function_type: str,
        input_field_names,
        output_field_names,
    ):
        self.name = name
        self.function_type = function_type
        self.input_field_names = input_field_names
        self.output_field_names = output_field_names


class FakeSchema:
    def __init__(self, auto_id: bool, enable_dynamic_field: bool):
        self.auto_id = auto_id
        self.enable_dynamic_field = enable_dynamic_field
        self.fields = []
        self.functions = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)

    def add_function(self, function):
        self.functions.append(function)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.collections = set()
        self.create_collection_calls = []
        self.insert_calls = []
        self.search_calls = []
        self.drop_collection_calls = []
        self.search_response = [[]]

    @staticmethod
    def create_schema(auto_id: bool, enable_dynamic_field: bool):
        return FakeSchema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
        )

    def list_collections(self):
        return list(self.collections)

    def drop_collection(self, collection_name: str):
        self.drop_collection_calls.append(collection_name)
        self.collections.discard(collection_name)

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, collection_name: str, schema, index_params, consistency_level: str):
        self.collections.add(collection_name)
        self.create_collection_calls.append(
            {
                "collection_name": collection_name,
                "schema": schema,
                "index_params": index_params,
                "consistency_level": consistency_level,
            }
        )

    def insert(self, collection_name: str, data):
        self.insert_calls.append(
            {
                "collection_name": collection_name,
                "data": data,
            }
        )
        return {"insert_count": len(data)}

    def search(
        self,
        collection_name: str,
        data,
        anns_field: str,
        filter,
        limit: int,
        output_fields,
        search_params,
    ):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "data": data,
                "anns_field": anns_field,
                "filter": filter,
                "limit": limit,
                "output_fields": output_fields,
                "search_params": search_params,
            }
        )
        return self.search_response


@pytest.fixture
def build_milvus_db(monkeypatch):
    monkeypatch.setattr(milvus_module, "MilvusClient", FakeMilvusClient)
    monkeypatch.setattr(milvus_module, "DataType", FakeDataType)
    monkeypatch.setattr(milvus_module, "Function", FakeFunction)
    monkeypatch.setattr(milvus_module, "FunctionType", FakeFunctionType)

    def _build(**kwargs):
        config = MilvusVectorDBConfig(
            collection_name="unit_test_collection",
            **kwargs,
        )
        return milvus_module.MilvusVectorDB(config)

    return _build


def test_create_client_uses_config_values(build_milvus_db):
    db = build_milvus_db(
        uri="http://localhost:19530",
        token="root:Milvus",
        db_name="vector_test",
    )

    assert db.client.kwargs == {
        "uri": "http://localhost:19530",
        "token": "root:Milvus",
        "db_name": "vector_test",
    }


def test_build_db_creates_expected_schema(build_milvus_db):
    db = build_milvus_db(vector_dim=4)

    collection_name = db.build_db()

    assert collection_name == "unit_test_collection"
    assert db.client.create_collection_calls

    create_call = db.client.create_collection_calls[0]
    schema = create_call["schema"]
    index_params = create_call["index_params"]

    assert create_call["collection_name"] == "unit_test_collection"
    assert create_call["consistency_level"] == "Bounded"
    assert schema.auto_id is False
    assert schema.enable_dynamic_field is False
    assert schema.fields == [
        {
            "field_name": "object_name",
            "datatype": "VARCHAR",
            "is_primary": True,
            "max_length": db.config.object_name_max_length,
        },
        {
            "field_name": "object_vector",
            "datatype": "FLOAT_VECTOR",
            "dim": 4,
        },
        {
            "field_name": "metadata",
            "datatype": "JSON",
        },
        {
            "field_name": "object_description",
            "datatype": "VARCHAR",
            "max_length": db.config.object_description_max_length,
            "enable_analyzer": True,
            "enable_match": True,
            "analyzer_params": {
                "tokenizer": "standard",
                "filter": ["lowercase"],
            },
        },
        {
            "field_name": "object_description_sparse",
            "datatype": "SPARSE_FLOAT_VECTOR",
        },
    ]
    assert len(schema.functions) == 1
    assert schema.functions[0].name == "object_description_bm25"
    assert schema.functions[0].function_type == "BM25"
    assert schema.functions[0].input_field_names == ["object_description"]
    assert schema.functions[0].output_field_names == ["object_description_sparse"]
    assert index_params.indexes == [
        {
            "field_name": "object_vector",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        },
        {
            "field_name": "object_description_sparse",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
        }
    ]


def test_normalize_vector_casts_values_to_float(build_milvus_db):
    db = build_milvus_db()

    assert db._normalize_vector([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_normalize_vector_rejects_empty_input(build_milvus_db):
    db = build_milvus_db()

    with pytest.raises(ValueError, match="Vectors must not be empty"):
        db._normalize_vector([])


def test_normalize_records_supports_single_record_input(build_milvus_db):
    db = build_milvus_db()

    records = db._normalize_records(
        object_name="object_alpha",
        object_vector=[1, 0, 0, 0],
        metadata={"kind": "alpha"},
        object_description="Alpha object.",
        records=None,
    )

    assert records == [
        {
            "object_name": "object_alpha",
            "object_vector": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"kind": "alpha"},
            "object_description": "Alpha object.",
        }
    ]


def test_normalize_records_supports_batch_input(build_milvus_db):
    db = build_milvus_db()

    records = db._normalize_records(
        object_name=None,
        object_vector=None,
        metadata=None,
        object_description="",
        records=[
            {
                "object_name": "object_alpha",
                "object_vector": [1, 0],
            },
            {
                "object_name": "object_beta",
                "object_vector": [0, 1],
                "metadata": {"kind": "beta"},
                "object_description": "Beta object.",
            },
        ],
    )

    assert records == [
        {
            "object_name": "object_alpha",
            "object_vector": [1.0, 0.0],
            "metadata": {},
            "object_description": "",
        },
        {
            "object_name": "object_beta",
            "object_vector": [0.0, 1.0],
            "metadata": {"kind": "beta"},
            "object_description": "Beta object.",
        },
    ]


def test_validate_vector_dim_raises_for_wrong_dimension(build_milvus_db):
    db = build_milvus_db(vector_dim=4)

    with pytest.raises(ValueError, match="Expected vectors with dimension 4, received 3"):
        db._validate_vector_dim([1.0, 2.0, 3.0])


def test_insert_auto_builds_collection_and_sets_vector_dim(build_milvus_db):
    db = build_milvus_db(vector_dim=None)

    result = db.insert(
        object_name="object_alpha",
        object_vector=[1, 0, 0, 0],
        metadata={"kind": "alpha"},
        object_description="Alpha object.",
    )

    assert result == {"insert_count": 1}
    assert db.config.vector_dim == 4
    assert db.collection_name in db.client.collections
    assert len(db.client.create_collection_calls) == 1
    assert db.client.insert_calls[0]["data"] == [
        {
            "object_name": "object_alpha",
            "object_vector": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"kind": "alpha"},
            "object_description": "Alpha object.",
        }
    ]


def test_insert_requires_existing_collection_when_auto_build_disabled(build_milvus_db):
    db = build_milvus_db(vector_dim=4)

    with pytest.raises(ValueError, match="does not exist. Run `build_db\\(\\)` first"):
        db.insert(
            object_name="object_alpha",
            object_vector=[1, 0, 0, 0],
            build_if_missing=False,
        )


def test_search_uses_defaults_and_formats_results(build_milvus_db):
    db = build_milvus_db(vector_dim=4)
    db.client.collections.add(db.collection_name)
    db.client.search_response = [
        [
            {
                "id": "object_alpha",
                "distance": 0.01,
                "entity": {
                    "metadata": {"category": "alpha"},
                    "object_description": "Alpha object.",
                },
            }
        ]
    ]

    results = db.search(query_vector=[1, 0, 0, 0], limit=2)

    assert results == [
        {
            "score": 0.01,
            "object_name": "object_alpha",
            "metadata": {"category": "alpha"},
            "object_description": "Alpha object.",
        }
    ]
    assert db.client.search_calls[0] == {
        "collection_name": "unit_test_collection",
        "data": [[1.0, 0.0, 0.0, 0.0]],
        "anns_field": "object_vector",
        "filter": None,
        "limit": 2,
        "output_fields": ["object_name", "metadata", "object_description"],
        "search_params": {
            "metric_type": "COSINE",
            "params": {},
        },
    }


def test_search_raises_when_collection_is_missing(build_milvus_db):
    db = build_milvus_db(vector_dim=4)

    with pytest.raises(ValueError, match="does not exist. Run `build_db\\(\\)` first"):
        db.search(query_vector=[1, 0, 0, 0])


def test_bm25_search_uses_description_sparse_index(build_milvus_db):
    db = build_milvus_db(vector_dim=4)
    db.client.collections.add(db.collection_name)
    db.client.search_response = [
        [
            {
                "id": "object_alpha",
                "distance": 0.2,
                "entity": {
                    "metadata": {"category": "alpha"},
                    "object_description": "Alpha keyword-rich document.",
                },
            }
        ]
    ]

    results = db.bm25_search(query_text="alpha keyword", limit=3)

    assert results == [
        {
            "score": 0.2,
            "object_name": "object_alpha",
            "metadata": {"category": "alpha"},
            "object_description": "Alpha keyword-rich document.",
        }
    ]
    assert db.client.search_calls[0] == {
        "collection_name": "unit_test_collection",
        "data": ["alpha keyword"],
        "anns_field": "object_description_sparse",
        "filter": None,
        "limit": 3,
        "output_fields": ["object_name", "metadata", "object_description"],
        "search_params": {
            "metric_type": "BM25",
            "params": {"drop_ratio_search": 0.0},
        },
    }


def test_bm25_search_rejects_empty_query(build_milvus_db):
    db = build_milvus_db(vector_dim=4)
    db.client.collections.add(db.collection_name)

    with pytest.raises(ValueError, match="`query_text` must not be empty"):
        db.bm25_search(query_text="   ")
