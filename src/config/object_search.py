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
    search_limit: int = 5
    all_objects_query: str = SPARQL_ALL_OBJECTS_TEMPLATE
    object_properties_query: str = SPARQL_OBJECT_PROPERTIES_TEMPLATE
    overwrite_db_collection:bool = True
