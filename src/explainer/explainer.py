from src.config.experiment import ApplicationInfo, ExplainerConfig, TTLConfig
from src.config.object_search import ObjectSearchConfig
from src.embeddings import Embeddings
from src.explainer.dependancy_graph import DependencyGraphRuntime
from src.llm import LLM
from src.utils.ontology_info_retriever import OntologyInfoRetriever
from src.vector_db import VectorDB

from src.utils.utils import time_wrapper

class Explainer:
    def __init__(self, 
                 kg_loc:str,
                 schema_loc:str,
                 schema_details_loc:str,
                 config:ExplainerConfig,
                 app_info:ApplicationInfo,
                 ttl_config:TTLConfig,
                 ) -> None:
        self.config:ExplainerConfig = config
        self.app_info = app_info
        
        self.llm = LLM(
            self.config.llm_type,
            "dspy",
            **self.config.llm_config
            )
        
        self.embedding = Embeddings(
            self.config.embedding_type,
            **self.config.embedding_config
        )
        
        self.db = VectorDB(
            self.config.vectordb_type,
            **self.config.vectordb_config
        )
        
        self.ontology_info = OntologyInfoRetriever(
            schema_details_loc,
            schema_path = schema_loc
        )
        
        self.dependancy_graph = DependencyGraphRuntime(
            kg_loc,
            app_info, 
            self.llm,
            self.embedding,
            self.db,
            ObjectSearchConfig(**self.config.object_search_config),
            ttl_config
        )
        
    def format_schema(self) -> str: 
        return self.ontology_info.format_schema_prompt()
    
    @time_wrapper
    def request(self, user_query:str):
        user_query = user_query.strip()
        application_context = (self.app_info.description or "").strip()
        self.dependancy_graph.user_query_to_requirements(
            user_query,
            schema_context=self.format_schema(),
            application_context=application_context,
            )
        
        return self.dependancy_graph.process_dependancy_graph(
            schema_context=self.format_schema(),
            application_context=application_context,
        )
