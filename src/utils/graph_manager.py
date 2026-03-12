from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF
import pandas as pd
from functools import partial
from src.utils.utils import regex_add_strings

logger = logging.getLogger(__name__)

LITERAL_MAP: Dict[str, str] = {
    "type.string": "http://www.w3.org/2001/XMLSchema#string",
    "type.text": "http://www.w3.org/2001/XMLSchema#string",
    "type.datetime": "http://www.w3.org/2001/XMLSchema#dateTime",
    "type.integer": "http://www.w3.org/2001/XMLSchema#int",
    "type.int": "http://www.w3.org/2001/XMLSchema#int",
    "type.float": "http://www.w3.org/2001/XMLSchema#float",
    "type.boolean": "http://www.w3.org/2001/XMLSchema#boolean",
}

# TEMPLATE_HEADER = """
#         PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#         PREFIX owl: <http://www.w3.org/2002/07/owl#>
#         PREFIX ep: <http://linkedu.eu/dedalo/explanationPattern.owl#>
#         PREFIX eo: <https://purl.org/heals/eo#>
#         PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
#         PREFIX dc: <http://purl.org/dc/elements/1.1/>
#         PREFIX food: <http://purl.org/heals/food/>
#         PREFIX prov: <http://www.w3.org/ns/prov#>
#         PREFIX provone: <http://purl.org/provone#>
#         PREFIX sio:<http://semanticscience.org/resource/>
#         """
        
TEMPLATE_TAIL = """
        \n 
        """

def is_inv_rel(rel: str) -> bool:
    """Check if a relation is an inverse relation."""
    return rel.endswith("#R")


def get_inv_rel(rel: str) -> str:
    """Get the inverse of a relation, or vice-versa."""
    if is_inv_rel(rel):
        return rel[:-2]  # Remove '#R'
    return f"{rel}#R"


def get_readable_class(cls: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Get a readable name for a class."""
    if schema and cls in schema["classes"] and "description" in schema["classes"][cls]:
        return schema["classes"][cls]["description"]
    return cls.split(".")[-1]


def get_readable_relation(rel: str, schema: Optional[Dict[str, Any]] = None) -> str:
    """Get a readable name for a relation."""
    if (
        schema
        and rel in schema["relations"]
        and "description" in schema["relations"][rel]
    ):
        return schema["relations"][rel]["description"]
    return rel.split(".")[-1]


def get_reverse_relation(rel: str, schema: Dict[str, Any]) -> Optional[str]:
    """Get the reverse relation from the schema."""
    return schema["relations"].get(rel, {}).get("reverse")


def get_reverse_readable_relation(rel: str, schema: Dict[str, Any]) -> Optional[str]:
    """Get the readable name of the reverse relation."""
    rev_rel = get_reverse_relation(rel, schema)
    if rev_rel and rev_rel in schema["relations"]:
        return schema["relations"][rev_rel].get("description")
    return None


def get_nodes_by_class(
    nodes: List[Dict[str, Any]], cls: str, except_nid: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Get all nodes of a specific class, with optional exceptions."""
    if except_nid is None:
        except_nid = []
    return [n for n in nodes if n["class"] == cls and n["nid"] not in except_nid]


def get_non_literals(
    nodes: List[Dict[str, Any]], except_nid: Optional[Set[int]] = None
) -> List[Dict[str, Any]]:
    """Get all nodes that are not literals."""
    if except_nid is None:
        except_nid = set()
    return [
        n
        for n in nodes
        if n["nid"] not in except_nid and not n["class"].startswith("type.")
    ]

def validate_namespaces(ns: dict):
    """
    Validates the structure of the namespace dictionary.
    R: validate_namespaces
    """
    if not isinstance(ns, dict) or not ns:
        raise ValueError("Namespaces must be a non-empty dictionary.")
    if any(not k for k in ns.keys()):
        raise ValueError("All namespace prefixes must have non-empty names.")
    bad_vals = [k for k, v in ns.items() if not (isinstance(v, str) and v)]
    if bad_vals:
        raise ValueError(f"All namespace IRIs must be non-empty strings. Offenders: {', '.join(bad_vals)}")
    logger.debug("Namespaces validated successfully.")
    return True

def make_ttl_namespace(yaml_config: dict) -> dict:
    """
    Creates a namespace dictionary from the YAML config.
    R: make_ttl_namespace
    """
    namespaces = {}
    for item in yaml_config.get('ttl', {}).get('prefixes', []):
        namespaces[item['name']] = item['uri']
    validate_namespaces(namespaces)
    return namespaces

def curie(x: str, ns: dict, default_prefix: Optional[str] = None, allow_bare: bool = False) -> str:
    """
    Expands a CURIE (e.g., "rdfs:label") into a full IRI.
    R: curie
    """
    if not isinstance(x, str):
        raise TypeError(f"Input must be a string, got {type(x)}")
    
    # R: if (x == "a") ...
    if x == "a":
        return str(RDF.type)
        
    # R: if (grepl("^(https?|urn):", x))
    if x.startswith(("http:", "https:", "urn:")):
        return x
        
    # R: if (grepl(":", x))
    if ":" in x:
        try:
            prefix, local = x.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid CURIE format: {x}")
            
        if not local:
            raise ValueError(f"Empty local part in CURIE: {x}")
        if prefix not in ns:
            raise ValueError(f"Unknown prefix in CURIE: {x}")
        return ns[prefix] + local
        
    # R: if (!is.null(default_prefix))
    if default_prefix:
        if default_prefix not in ns:
            raise ValueError(f"default_prefix '{default_prefix}' not found in ns")
        return ns[default_prefix] + x
        
    if allow_bare:
        return x
        
    raise ValueError(f"Not a CURIE (no ':') and not a full IRI: {x}")

def reverse_curie(iri: str, ns: dict) -> str:
    """
    Converts a full IRI back to a CURIE using the provided namespaces.
    R: reverse_curie
    """
    for prefix, uri in ns.items():
        if iri.startswith(uri):
            local_part = iri[len(uri):]
            return f"{prefix}:{local_part}"
    return iri  # Return as-is if no matching prefix found

def add_to_graph_func(s: str, p: str, o: str, g: Graph, namespaces: dict, 
                      literal: bool = False, lang: Optional[str] = None, dtype: Optional[str] = None):
    """
    Adds a triple to the rdflib Graph, handling CURIE expansion.
    R: add_to_graph
    """
    try:
        s_uri = URIRef(curie(s, namespaces))
        p_uri = URIRef(curie(p, namespaces))
        
        o_obj: Union[URIRef, Literal]
        if literal:
            if dtype:
                o_obj = Literal(o, datatype=URIRef(curie(dtype, namespaces)))
            elif lang:
                o_obj = Literal(o, lang=lang)
            else:
                o_obj = Literal(o)
        else:
            o_obj = URIRef(curie(o, namespaces))
            
        g.add((s_uri, p_uri, o_obj))
        
    except Exception as e:
        logger.error(f"Failed to add triple: ({s}, {p}, {o}). Error: {e}")
        
def resolve_curie(
    x: str, 
    ns: dict, 
    default_prefix: Optional[str] = None, 
    allow_bare: bool = False) -> str:
    
    """
    Resolves a CURIE or returns the input if it's already a full IRI.
    R: resolve_curie
    """
    try:
        return curie(x, ns, default_prefix, allow_bare)
    except ValueError:
        return x
    
# --- 4. Query Graph ---

def query_func(g: Graph, sparql_query: str, *args) -> pd.DataFrame:
    """
    Executes a SPARQL query and returns a pandas DataFrame.
    R: query_func
    """
    try:
        # R: query <- sprintf(sparql_query_temp_get_objects, ...)
        if args:
            query = sparql_query % args
        else:
            query = sparql_query
        
        results = g.query(query)
        
        # Convert results to a pandas DataFrame
        # R: return(query_results)
        data = []
        for row in results:
            data.append({str(var): str(val) for var, val in row.asdict().items()})
        
        if not data:
            return pd.DataFrame(columns=[str(v) for v in results.vars])
            
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to execute SPARQL query: {e}")
        return pd.DataFrame()

class GraphManager:
    """
    A class to hold graph state, config, and helper functions,
    replacing the R environment `graph_func`.
    """
    def __init__(self, config: dict, graph_file: Optional[str] = None):
        logger.info("Initializing GraphManager...")
        self.config = config
        self.graph = Graph()
        if graph_file:
            self.graph.parse(graph_file, format="turtle")
        logger.info(f"Graph loaded with {len(self.graph)} triples.")
        
        self.config['namespaces'] = make_ttl_namespace(self.config)
        
        # Create a partial function, same as R's `partial()`
        # R: graph_func$add_to_graph <- partial(add_to_graph, ...)
        self.add_to_graph = partial(add_to_graph_func, 
                                    g=self.graph, 
                                    namespaces=self.config['namespaces'])

        self.reverse_curie = partial(reverse_curie, ns=self.config['namespaces'])

    def query(self, sparql_query: str, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get('add_header_tail', True):
            sparql_query = self.add_sparql_header_tail(sparql_query)
        
        results = query_func(self.graph, sparql_query, *args)
        
        if kwargs.get('resolve_curie', False):
            for col in results.columns:
                results[col] = results[col].apply(self.reverse_curie)

        return results
    
    def add_sparql_header_tail(self, txt):
        header = ""
        for item_name, item_uri in self.config['namespaces'].items():
            header += f"PREFIX {item_name}: <{item_uri}> \n"
            
        header += "\n"
        return f"{header} {txt} {TEMPLATE_TAIL}"
    
    @staticmethod
    def legal_class(_class: str) -> bool:
        """Check if a class is a legal starting point (not a literal)."""
        return not _class.startswith("type.")
    
    @staticmethod
    def legal_relation(rel: str) -> bool:
        """Placeholder for relation filtering logic, if any."""
        # You can add logic here to filter out specific relations
        return True


    def literal_for_class(self, cls: str) -> Tuple[list, list, list]:
        """
        Given a class name find all the literal relationship of that class and return a Literal with appropriate datatype.
        by sparql query on the graph.
        """
        
        # Example implementation (you may need to adjust based on your graph structure)
        
        query = """SELECT ?obj ?relation ?datatype WHERE {
                    ?obj a <{cls}> .
                    ?obj ?relation ?datatype .
                }
                """
        
        query = regex_add_strings(query, cls=cls)
        results = self.query(query)
        if results.empty:
            return ([], [], [])
        results['is_literal'] = results['datatype'].apply(lambda dt: '@' in str(dt))
        literal_rows = results[results['is_literal']]
        literal_rows = literal_rows.groupby('relation').agg({'obj': list, 'datatype': list}).reset_index()
        obj_list = []
        relation_list = []
        datatype_list = []
        if literal_rows.empty:
            return ([], [], [])
        for _, row in literal_rows.iterrows():
            obj_list.append(row['obj'])
            relation_list.append(row['relation'])
            datatype_list.append(row['datatype'])

        return obj_list, relation_list, datatype_list
    
    def resolve_curie(
        self,
        x: str, 
        default_prefix: Optional[str] = None, 
        allow_bare: bool = False) -> str:
        
        """
        Resolves a CURIE or returns the input if it's already a full IRI.
        R: resolve_curie
        """
        
        ns = self.config['namespaces']
        return resolve_curie(x, ns, default_prefix, allow_bare)