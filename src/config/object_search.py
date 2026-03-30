from typing import Any, Dict

from pydantic import BaseModel, Field

SPARQL_ALL_OBJECTS_TEMPLATE = """
SELECT DISTINCT ?object WHERE {
    ?object ?predicate ?value .
    FILTER(isIRI(?object))
}
ORDER BY ?object
"""

SPARQL_OBJECT_PROPERTIES_TEMPLATE = """
SELECT DISTINCT ?predicate ?value WHERE {
    <{object_uri}> ?predicate ?value .
}
ORDER BY ?predicate ?value
"""


class ObjectSearchConfig(BaseModel):
    collection_name: str = "calibration"
    embedding_type: str = "lmstudio"
    embedding_library: str = "langchain"
    embedding_config: Dict[str, Any] = Field(default_factory=dict)
    vector_db_type: str = "milvus"
    vector_db_config: Dict[str, Any] = Field(default_factory=dict)
    search_limit: int = 5
    all_objects_query: str = SPARQL_ALL_OBJECTS_TEMPLATE
    object_properties_query: str = SPARQL_OBJECT_PROPERTIES_TEMPLATE
