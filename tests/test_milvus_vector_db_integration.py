import socket
import sys
import uuid
from pathlib import Path
from time import sleep

import pytest

src_path = Path(__file__).resolve().parent.parent

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def _milvus_is_available(host: str = "127.0.0.1", port: int = 19530) -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


@pytest.fixture
def milvus_db():
    pytest.importorskip("pymilvus")

    if not _milvus_is_available():
        pytest.skip("Milvus is not running on localhost:19530")

    from src.vector_db import VectorDB

    collection_name = f"test_object_store_{uuid.uuid4().hex[:8]}"
    db = VectorDB(
        "milvus",
        uri="http://localhost:19530",
        collection_name=collection_name,
        vector_dim=4,
    )

    try:
        yield db
    finally:
        if collection_name in db.client.list_collections():
            db.client.drop_collection(collection_name=collection_name)


def _sample_records():
    return [
        {
            "object_name": "object_alpha",
            "object_vector": [1.0, 0.0, 0.0, 0.0],
            "metadata": {"category": "alpha", "rank": 1},
            "object_description": "Primary alpha test object.",
        },
        {
            "object_name": "object_beta",
            "object_vector": [0.0, 1.0, 0.0, 0.0],
            "metadata": {"category": "beta", "rank": 2},
            "object_description": "Secondary beta test object.",
        },
    ]


def test_milvus_build_and_insert_integration(milvus_db):
    collection_name = milvus_db.build_db(overwrite=True)
    assert collection_name == milvus_db.collection_name

    insert_result = milvus_db.insert(records=_sample_records())

    assert insert_result is not None
    assert insert_result["insert_count"] == 2
    assert set(insert_result["ids"]) == {"object_alpha", "object_beta"}


def test_milvus_search_integration(milvus_db):
    milvus_db.build_db(overwrite=True)
    milvus_db.insert(records=_sample_records())

    # This flush is kept in the test only so search is deterministic immediately after setup.
    if hasattr(milvus_db.client, "flush"):
        milvus_db.client.flush(collection_name=milvus_db.collection_name)

    search_results = []
    for _ in range(5):
        search_results = milvus_db.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            limit=2,
        )
        if search_results:
            break
        sleep(0.5)

    assert len(search_results) >= 1

    top_result = search_results[0]
    assert top_result["object_name"] == "object_alpha"
    assert top_result["metadata"]["category"] == "alpha"
    assert top_result["metadata"]["rank"] == 1
    assert top_result["object_description"] == "Primary alpha test object."
    assert "score" in top_result
