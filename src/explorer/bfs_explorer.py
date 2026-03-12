import pandas as pd

from src.utils.utils import time_wrapper, regex_add_strings, generate_hashed_filename
from src.explorer.executable_program import ExecutableProgram
from src.utils.graph_manager import GraphManager
from rdflib import Graph, URIRef, Literal, RDF, RDFS
from collections import defaultdict
import json
import os
import pickle
import random
import logging
from icecream import ic
from typing import DefaultDict, List, Dict, Any, Optional, Set, Tuple
from tqdm import tqdm
import time
import yaml
import dycomutils as common_utils

logger = logging.getLogger(__name__)

# Get all objects of a class
SPARQL_OBJ_OF_CLASS_TEMPLATE = """SELECT DISTINCT ?value WHERE {
                            ?value a <{class_uri}> .
                            }"""
                            
# Get all propertiess of a object
SPARQL_PROP_OF_OBJ_TEMPLATE = """
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT DISTINCT ?p ?o ?pe ?po
                WHERE {
                    # 1. Start with the properties of the target object
                    <{obj_uri}> ?p ?o .
                    
                    # 2. Check if the object is a prov:Collection
                    # This pattern only binds ?isCollection if <{obj_uri}> is a prov:Collection
                    OPTIONAL {
                        <{obj_uri}> rdf:type prov:Collection .
                        BIND(TRUE AS ?isCollection)
                    }
                    
                    OPTIONAL {
                        FILTER (bound(?isCollection))
                        <{obj_uri}> prov:hadMember ?member .
                        ?member ?pe ?po .
                    }
                }"""
                
# Find objects that have a specific relationship to a given object
SPARQL_FIND_BY_OBJ_REL_TEMPLATE = """SELECT DISTINCT ?value WHERE {
    <{obj_uri}> <{relation_uri}> ?value .
    }"""
      
# Find objects that have a specific property value
SPARQL_FIND_BY_PROP_VAL_TEMPLATE = """SELECT DISTINCT ?value WHERE {
    ?value <{relation_uri}> ?prop .
    FILTER(CONTAINS(STR(?prop), "{prop_value}"))
}"""


def start_to_end_path_processing(
    path: List[str], 
    graph_manager: GraphManager,
    str_representation: str,
    temp_folder: str = "tmp/programs"
    ) -> ExecutableProgram:
    
    logger.debug(f"Converting path to graph: {str_representation}")

    # Create nodes
    str_query = "SELECT distinct ?value where {\n" 
    for i in range(1, len(path) -1, 2):
        # Resolve CURIE to get full URI for class lookup
        pred = path[i] 
        obj = path[i + 1]
        
        if len(path) == 3:
            str_query += "  <{obj}>"+f" {pred} "+" ?value  .\n"
            break
        
        if i == 1:
            str_query += "  <{obj}>"+f" {pred} "+f"?a{i}  .\n"
        elif i == len(path) - 2:
            str_query += f"  ?a{i-2}"+f" {pred} "+" ?value  .\n"
        else:
            str_query += f"  ?a{i-2}"+f" {pred} "+f"?a{i}  .\n"
    str_query += "}"

    #get objects for the class path
    query_df = graph_manager.query(
        regex_add_strings(
            SPARQL_OBJ_OF_CLASS_TEMPLATE,
            class_uri=graph_manager.resolve_curie(path[0])
        )
    )

    objs = list(set(query_df['value'].to_list()))
    if len(objs) == 0:
        logger.debug(f"No objects found for class: {path[0]} in path: {str_representation}")
        return None
    
    example_output = None
    example_query = None
    for obj in random.sample(objs, len(objs) ):
        example_query = regex_add_strings(
            str_query,
            obj=obj
        )

        #print(example_query)
        print("Executing example query...")
        start_t = time.time()
        example_output = graph_manager.query(example_query)
        end_t = time.time()
        print("end execution.")
        print(f"Query took {end_t - start_t} seconds")
        if not example_output.empty:
            break
        
    if example_output is None or example_output.empty or example_query is None:
        logger.debug(f"No results for path: {str_representation}")
        return None

    question = f"What are the values obtained by traversing the path: {str_representation}?"

    return ExecutableProgram(
        program_id=f"explore_path_{str_representation}",
        name=f"Explore Path {str_representation}",
        description=question,
        input_spec={"obj": "The URI of the starting object."},
        output_spec={"value": "The resulting values from the path traversal."},
        code=graph_manager.add_sparql_header_tail(
            str_query
        ),
        solves=f"What are the values obtained by traversing the path: {str_representation}?",
        example_usage=example_query,
        example_output=example_output.head(10),
        tags=["path-level", *path],
        metadata={
            "path": path
            }
    )
    
def end_to_start_path_processing(
    path: List[str], 
    graph_manager: GraphManager,
    str_representation: str,
    temp_folder: str = "tmp/programs"
    ) -> ExecutableProgram:
    
    logger.debug(f"Converting path to graph reversal: {str_representation}")

    # Create nodes
    str_query = "SELECT distinct ?value where {\n" 
    for i in range(1, len(path) -1, 2):
        # Resolve CURIE to get full URI for class lookup
        pred = path[i] 
        obj = path[i + 1]
        
        if len(path) == 3:
            str_query += "  ?value"+f" {pred} "+"  <{obj}> .\n"
            break
        
        if i == 1:
            str_query += "  ?value"+f" {pred} "+f"?a{i}  .\n"
        elif i == len(path) - 2:
            str_query += f"  ?a{i-2}"+f" {pred} "+" <{obj}>  .\n"
        else:
            str_query += f"  ?a{i-2}"+f" {pred} "+f"?a{i}  .\n"
    str_query += "}"

    #get objects for the class path
    query_df = graph_manager.query(
        regex_add_strings(
            SPARQL_OBJ_OF_CLASS_TEMPLATE,
            class_uri=graph_manager.resolve_curie(path[-1])
        )
    )

    objs = list(set(query_df['value'].to_list()))
    if len(objs) == 0:
        logger.debug(f"No objects found for class: {path[-1]} in path: {str_representation}")
        return None
    
    example_output = None
    example_query = None
    for obj in random.sample(objs, len(objs) ):
        example_query = regex_add_strings(
            str_query,
            obj=obj
        )

        #print(example_query)
        print("Executing example query...")
        start_t = time.time()
        example_output = graph_manager.query(example_query)
        end_t = time.time()
        print("end execution.")
        print(f"Query took {end_t - start_t} seconds")
        if not example_output.empty:
            break
        
    if example_output is None or example_output.empty or example_query is None:
        logger.debug(f"No results for path: {str_representation}")
        return None

    question = f"What objects leads to this path: {str_representation}?"

    return ExecutableProgram(
        program_id=f"explore_path_{str_representation}_reversed",
        name=f"Explore Path {str_representation} reversed",
        description=question,
        input_spec={"obj": "The URI of the ending object."},
        output_spec={"value": "The resulting values from the path traversal."},
        code=graph_manager.add_sparql_header_tail(
            str_query
        ),
        solves=f"What are the values obtained by traversing the path reversed: {str_representation}?",
        example_usage=example_query,
        example_output=example_output.head(10),
        tags=["path-level", *path],
        metadata={
            "path": path
            }
    )
    
def function_path_processing(
    path: List[str], 
    graph_manager: GraphManager,
    str_representation: str,
    function:str,
    temp_folder: str = "tmp/programs"
    ) -> ExecutableProgram:
    
    logger.debug(f"Converting path to graph reversal: {str_representation}")

    # Create nodes
    str_query = "SELECT distinct ?value where {\n" 
    for i in range(1, len(path) -1, 2):
        # Resolve CURIE to get full URI for class lookup
        pred = path[i] 
        obj = path[i + 1]
        
        if len(path) == 3:
            str_query += "  ?value"+f" {pred} "+"  <{obj}> .\n"
            break
        
        if i == 1:
            str_query += "  ?value"+f" {pred} "+f"?a{i}  .\n"
        elif i == len(path) - 2:
            str_query += f"  ?a{i-2}"+f" {pred} "+" <{obj}>  .\n"
        else:
            str_query += f"  ?a{i-2}"+f" {pred} "+f"?a{i}  .\n"
    str_query += "}"

    #get objects for the class path
    query_df = graph_manager.query(
        regex_add_strings(
            SPARQL_OBJ_OF_CLASS_TEMPLATE,
            class_uri=graph_manager.resolve_curie(path[-1])
        )
    )

    objs = list(set(query_df['value'].to_list()))
    if len(objs) == 0:
        logger.debug(f"No objects found for class: {path[-1]} in path: {str_representation}")
        return None
    
    example_output = None
    example_query = None
    for obj in random.sample(objs, len(objs) ):
        example_query = regex_add_strings(
            str_query,
            obj=obj
        )

        #print(example_query)
        print("Executing example query...")
        start_t = time.time()
        example_output = graph_manager.query(example_query)
        end_t = time.time()
        print("end execution.")
        print(f"Query took {end_t - start_t} seconds")
        if not example_output.empty:
            break
        
    if example_output is None or example_output.empty or example_query is None:
        logger.debug(f"No results for path: {str_representation}")
        return None

    question = f"What objects leads to this path: {str_representation}?"

    return ExecutableProgram(
        program_id=f"explore_path_{str_representation}_reversed",
        name=f"Explore Path {str_representation} reversed",
        description=question,
        input_spec={"obj": "The URI of the ending object."},
        output_spec={"value": "The resulting values from the path traversal."},
        code=graph_manager.add_sparql_header_tail(
            str_query
        ),
        solves=f"What are the values obtained by traversing the path reversed: {str_representation}?",
        example_usage=example_query,
        example_output=example_output.head(10),
        tags=["path-level", *path],
        metadata={
            "path": path
            }
    )


@time_wrapper
def path_to_graph(
    path: List[str], 
    graph_manager: GraphManager,
    temp_folder: str = "tmp/programs"
    ) -> List[ExecutableProgram]:
        """
        Converts a single path (a list of node URIs) into a graph structure.
        """
        
        str_representation = "->".join(path)
        file_path:str = os.path.join(
            temp_folder, 
            generate_hashed_filename(
                str_representation,
                ".pkl"
                )
            )
        if os.path.exists(file_path):
            return common_utils.serialization.load_pickle(file_path)
    
        prog1 = start_to_end_path_processing(
                path=path,
                graph_manager=graph_manager,
                str_representation=str_representation,
                temp_folder=temp_folder
            )
        
        prog2 = end_to_start_path_processing(
                path=path,
                graph_manager=graph_manager,
                str_representation=str_representation,
                temp_folder=temp_folder
            )
        
        progs = [prog1, prog2]
        common_utils.serialization.save_pickle(
            progs, 
            file_path
            )
        return progs
    
def process_path(path: List[str], graph:GraphManager, temp_folder):# Only consider paths with at least one edge
    query_graph = path_to_graph(path, graph, temp_folder)
    if query_graph is not None:
        return query_graph, ' -> '.join(path)
    return None, None


class BFSExplorer:
    """
    Loads a pre-built RDF graph and its JSON schema to perform
    random schema-guided traversals (walks).
    """


    def __init__(self, 
                 kg_name: str, 
                 graph_manager: GraphManager, 
                 ontology_info_triples: pd.DataFrame,
                 parallel_execution: bool = False,
                 temp_folder: str = "tmp/programs"
                 ):
        self.kg_name: str = kg_name
        self.graph_manager: GraphManager = graph_manager
        self.parallel_execution: bool = parallel_execution
        self.ontology_info_triples: pd.DataFrame = ontology_info_triples
        self.temp_folder: str = temp_folder
        self.schema: Optional[Dict[str, Any]] = None
        self.schema_dr: Dict[str, Tuple[str, str]] = {}
        self.classes: Set[str] = set()

        # In-memory representation of the graph and schema
        self.out_relations_cls: DefaultDict[str, set] = defaultdict(set)
        self.in_relations_cls: DefaultDict[str, set] = defaultdict(set)
        self.cls_2_entid: DefaultDict[str, set] = defaultdict(set)
        self.entid_2_cls_ent: Dict[str, Dict[str, Any]] = {}
        self.literals_by_cls_rel: DefaultDict[Tuple[str, str], set] = defaultdict(set)
        
        self.all_program_Obj = []
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

    def load_graph_and_schema(
        self,
        schema_fpath: str,
        rdf_fpath: str,
        metadata_path: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Loads the JSON schema and the RDF graph file.
        It builds the in-memory representation needed for exploration.

        Args:
            schema_fpath: Path to the JSON schema file.
            rdf_fpath: Path to the RDF graph file (e.g., .nt, .ttl, .rdf).
            metadata_path: Path to a .pkl file for caching the processed data.
            use_cache: If True, try to load from metadata_path if it exists.
        """
 
        if use_cache and metadata_path and os.path.exists(metadata_path):
            logger.info(f"Loading cached processed data from {metadata_path}")
            with open(metadata_path, "rb") as f:
                processed = pickle.load(f)
                self.schema = processed["schema"]
                self.schema_dr = processed["schema_dr"]
                self.classes = processed["classes"]
                self.out_relations_cls = processed["out_relations_cls"]
                self.in_relations_cls = processed["in_relations_cls"]
                self.cls_2_entid = processed["cls_2_entid"]
                self.entid_2_cls_ent = processed["entid_2_cls_ent"]
                self.literals_by_cls_rel = processed["literals_by_cls_rel"]
            return

        logger.info(f"Processing schema from {schema_fpath}")

        # 1. Load Schema
        with open(schema_fpath, "r") as f:
            self.schema = json.load(f)

        if not self.schema:
            raise ValueError("Schema could not be loaded or is empty.")

        self.classes = set(self.schema.get("classes", []))

        for rel, rel_obj in self.schema.get("object_properties", {}).items():
            connections = rel_obj.get("connections", [])
            for conn in connections:
                domain = conn["domain"]
                range_ = conn["range"]

                self.schema_dr[rel] = (domain, range_)
                self.out_relations_cls[domain].add(rel)
                self.in_relations_cls[range_].add(rel)

        logger.info(f"Loading RDF graph from {rdf_fpath}...")

        # 2. Load RDF Graph
        g = Graph()
        try:
            g.parse(rdf_fpath)
        except Exception as e:
            logger.error(f"Failed to parse RDF file {rdf_fpath}: {e}")
            raise

        logger.info(f"Graph loaded with {len(g)} triples. Indexing entities...")

        # 3. Build in-memory indexes from the graph

        # Get RDFS.label, fall back to a common alt
        label_prop = RDFS.label

        # Index Entities and their labels
        for cls in tqdm(self.classes, desc="Indexing entities by class"):
            if cls.startswith("type."):  # Skip literal types
                continue

            try:
                cls_uri = URIRef(cls)
                for ent_uri in g.subjects(RDF.type, cls_uri):
                    if not isinstance(ent_uri, URIRef):
                        continue  # Skip blank nodes

                    ent_id_str = str(ent_uri)
                    self.cls_2_entid[cls].add(ent_id_str)

                    # Get label
                    label_lit = g.value(ent_uri, label_prop)
                    label_str = (
                        str(label_lit) if label_lit else ent_id_str.split("/")[-1]
                    )

                    self.entid_2_cls_ent[ent_id_str] = {"class": cls, "name": label_str}
            except Exception as e:
                logger.warning(f"Error indexing class {cls}: {e}")

        # Index Literals by (domain_class, relation)
        for rel, (domain, range_) in tqdm(
            self.schema_dr.items(), desc="Indexing literals"
        ):
            if not range_.startswith("type."):  # Skip non-literal ranges
                continue

            try:
                domain_uri = URIRef(domain)
                rel_uri = URIRef(rel)

                # Find all subjects of the domain type
                for s_uri in g.subjects(RDF.type, domain_uri):
                    # For each subject, get the literal objects for this relation
                    for o_lit in g.objects(s_uri, rel_uri):
                        if isinstance(o_lit, Literal):
                            self.literals_by_cls_rel[(domain, rel)].add(str(o_lit))
            except Exception as e:
                logger.warning(f"Error indexing literals for relation {rel}: {e}")

        logger.info("Finished processing graph and schema.")

        # 4. Save to cache if path provided
        if use_cache and metadata_path:
            logger.info(f"Saving processed data to cache at {metadata_path}")
            try:
                with open(metadata_path, "wb") as f:
                    pickle.dump(
                        {
                            "schema": self.schema,
                            "schema_dr": self.schema_dr,
                            "classes": self.classes,
                            "out_relations_cls": self.out_relations_cls,
                            "in_relations_cls": self.in_relations_cls,
                            "cls_2_entid": self.cls_2_entid,
                            "entid_2_cls_ent": self.entid_2_cls_ent,
                            "literals_by_cls_rel": self.literals_by_cls_rel,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except Exception as e:
                logger.warning(f"Failed to write to cache file {metadata_path}: {e}")
                         
    def explore_object_of_class(self):
        
        # example  execution of SPARQL query
        class_Name = self.graph_manager.resolve_curie('provone:Execution')
        run_query = regex_add_strings(SPARQL_OBJ_OF_CLASS_TEMPLATE, class_uri=class_Name)
        
        question_df = self.graph_manager.query(run_query)
        
        exe1 = ExecutableProgram(
            program_id="explore_object_of_class",
            name="Explore Objects of Class",
            description="Explores objects of a given class in the RDF graph.",
            input_spec={"class_uri": "The URI of the class to explore."},
            output_spec={"objects": "A list of objects belonging to the specified class."},
            code=SPARQL_OBJ_OF_CLASS_TEMPLATE,
            solves="What are all the objects of a given class?",
            example_usage=regex_add_strings(SPARQL_OBJ_OF_CLASS_TEMPLATE, class_uri=class_Name),
            example_output=question_df.head(10),
            tags=["class-level"]
        )

        
        # example  execution of SPARQL query
        obj_Name = self.graph_manager.resolve_curie('http://testwebsite/testProgram#AI_Task-information_extractor')
        run_query = regex_add_strings(SPARQL_PROP_OF_OBJ_TEMPLATE, obj_uri=obj_Name)
        
        question_df = self.graph_manager.query(run_query)
        print(question_df.head(10))
        
        exe2 = ExecutableProgram(
            program_id="explore_attr_of_object",
            name="Explore Attributes of Object",
            description="Explores Attributes of a given object in the RDF graph.",
            input_spec={"obj_uri": "The URI of the object to explore."},
            output_spec={"attributes": "A list of attributes belonging to the specified object."},
            code=SPARQL_PROP_OF_OBJ_TEMPLATE,
            solves="What are all the ATTRIBUTES AND VALUES of a given object?",
            example_usage=regex_add_strings(SPARQL_PROP_OF_OBJ_TEMPLATE, obj_uri=obj_Name),
            example_output=question_df.head(10),
            tags=["class-level"]
        )
        
        return [exe1, exe2]
    
    def explore_literal_paths(self):
        """
        Generates executable programs for finding objects by property value and
        finding property values for a given object.
        """
        programs = []
        
        for c in self.classes:
            if not self.graph_manager.legal_class(c):
                continue
            
            obj_list, relation_list, datatype_list = self.graph_manager.literal_for_class(
                self.graph_manager.resolve_curie(c)
                )
            if not obj_list or not relation_list or not datatype_list:
                continue
            
            for data in zip(obj_list, relation_list, datatype_list):
                objs, relations, raw_data = data
                if not self.graph_manager.legal_relation(relations):
                    continue

                example_output = None
                example_query = None
                for ob in objs:
                    example_query = regex_add_strings(
                            SPARQL_FIND_BY_OBJ_REL_TEMPLATE,
                            obj_uri=ob,
                            relation_uri=relations
                        )
                    example_output = self.graph_manager.query(example_query)
                    if example_output.empty:
                        continue
                    else:
                        break
                    
                if example_output is None or example_output.empty or example_query is None:
                    continue
                
                logging.info(regex_add_strings(
                            SPARQL_FIND_BY_OBJ_REL_TEMPLATE,
                            relation_uri=relations
                        ))
                
                p = ExecutableProgram(
                    program_id="explore_object_of_class {c} | find by object uri value | relation:{relation}".format(
                        c=self.graph_manager.reverse_curie(c),
                        relation=self.graph_manager.reverse_curie(relations)
                        ),
                    name="Explore Objects of Class",
                    description="For a given object with uri of class {c}, find all {relation} values.".format(
                        c=self.graph_manager.reverse_curie(c),
                        relation=self.graph_manager.reverse_curie(relations)
                        ),
                    input_spec={"obj_uri": "The URI of the object of interest."},
                    output_spec={"relation_uri": "Relation of interest."},
                    code=self.graph_manager.add_sparql_header_tail(
                        regex_add_strings(
                            SPARQL_FIND_BY_OBJ_REL_TEMPLATE,
                            relation_uri=relations
                        )
                    ),
                    solves="What are all the objects of a given class?",
                    example_usage=example_query,
                    example_output=example_output.head(10),
                    tags=["object-level", "from-object"]
                )

                programs.append(p)
                
                example_output = None
                example_query = None
                for sub in raw_data:
                    example_query = regex_add_strings(
                            SPARQL_FIND_BY_PROP_VAL_TEMPLATE,
                            prop_value=sub,
                            relation_uri=relations  
                        )
                    example_output = self.graph_manager.query(example_query)
                    
                    if example_output.empty:
                        continue
                    else:
                        break
                    
                if example_output is None or example_output.empty or example_query is None:
                    continue
                
                p = ExecutableProgram(
                    program_id="explore_object_of_class {c} | find by prop value | relation:{relation}".format(
                        c=self.graph_manager.reverse_curie(c),
                        relation=self.graph_manager.reverse_curie(relations)
                        ),
                    name="Explore Objects of Class",
                    description="For a given prop value of class {c}, find all {relation} prop value of the object.".format(
                        c=self.graph_manager.reverse_curie(c),
                        relation=self.graph_manager.reverse_curie(relations)
                        ),
                    input_spec={"prop_uri": "The URI of the prop of interest."},
                    output_spec={"relation_uri": "Relation of interest."},
                    code=self.graph_manager.add_sparql_header_tail(
                        regex_add_strings(
                            SPARQL_FIND_BY_OBJ_REL_TEMPLATE,
                            relation_uri=relations
                        )
                    ),
                    solves="What are all the objects of a given class?",
                    example_usage=example_query,
                    example_output=example_output,
                    tags=["object-level", 'from-prop']
                )

                programs.append(p)

        return programs

    def explore_workflow_graph(self, save_loc:str):

        # All object of class
        self.all_program_Obj.extend(self.explore_object_of_class())
        print(f"Total programs after object of class: {len(self.all_program_Obj)}")

        # Explore class methods
        self.all_program_Obj.extend(self.explore_literal_paths())
        print(f"Total programs after literal paths: {len(self.all_program_Obj)}")
        
        # Methods to object
        self.all_program_Obj.extend(self.generate_queries_from_paths())
        print(f"Total programs after generating queries from paths: {len(self.all_program_Obj)}")
        
        common_utils.serialization.save_pickle(
            self.all_program_Obj,
            save_loc
        )
    
    @time_wrapper
    def breadth_first_search(self, start_class: str, entity_length: int = 7) -> List[List[str]]:
        """
        Performs a schema-aware breadth-first search to find all simple paths.
        This search explores the ontology (schema) rather than the instance data.

        Args:
            start_class_uri: The URI of the starting class (can be a CURIE).

        Returns:
            A list of all simple paths, where each path is a list of class URIs.
        """
        from collections import deque

        if not isinstance(start_class, str) or not start_class.strip():
            raise ValueError("start_class must be a non-empty string.")


        if start_class not in self.classes:
            logger.warning(f"Start class {start_class} not found in the schema.")
            return []

        # The queue will store tuples of (current_class_uri, path_list)
        # The path is stored with original CURIEs/URIs for consistency
        queue = deque([(start_class, [start_class])])
        all_paths = []

        while queue:
            current_class, current_path = queue.popleft()

            # Find neighbors using the pre-indexed schema relations
            
            # 1. Outgoing relations (current_class is the domain)
            if current_class in self.out_relations_cls:
                for relation in self.out_relations_cls[current_class]:
                    # The range is the neighbor class
                    _, neighbor_class = self.schema_dr[relation]
                    if neighbor_class not in [c for c in current_path]:
                        new_path = current_path + [relation] + [self.graph_manager.reverse_curie(neighbor_class)]
                        all_paths.append(new_path)
                        
                        if len(new_path) // 2 < entity_length:
                            queue.append((neighbor_class, new_path))

        
        logger.info(f"BFS from '{start_class}' found {len(all_paths)} simple schema paths.")
        return all_paths
               
    
    def generate_queries_from_paths(self):
        """
        Generates a graph query for every simple path found via BFS from each class.
        """
            
        collected_graphs = {}           
        for c in self.classes:
            if not self.graph_manager.legal_class(c):
                continue
            
            # We can start BFS from the class URI itself to find paths in the schema
            paths = self.breadth_first_search(c)
            
            params = {k:[v, self.graph_manager, self.temp_folder] for k,v in enumerate(paths)}
            
            if len(params) == 0:
                continue
                
            if  self.parallel_execution:
                for _,v in common_utils.concurrancy.concurrent_dict_execution(
                    process_path,
                    params= params,
                    num_max_workers=20
                ):
                    if v[0] is not None and v[1] is not None:
                        if v[1] not in collected_graphs:
                            collected_graphs[v[1]] = v[0]
            else:
                for _, path in params.items():
                    query_graph, path_str = process_path(
                        path[0], 
                        self.graph_manager,
                        self.temp_folder
                        )
                    if query_graph is not None and path_str is not None:
                        if path_str not in collected_graphs:
                            collected_graphs[path_str] = query_graph

        logger.info(f"Generated {len(collected_graphs)} query graphs from all class paths.")
        return list(collected_graphs.values())
    