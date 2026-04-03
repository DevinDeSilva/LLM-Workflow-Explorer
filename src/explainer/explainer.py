from src.config.object_search import ObjectSearchConfig
from src.llm import LLM
from src.embeddings import Embeddings
from src.vector_db import VectorDB
from src.explainer.dependancy_graph import DependancyGraph
from src.config.experiment import ApplicationInfo, ExplainerConfig, TTLConfig
from src.utils.ontology_info_retriever import OntologyInfoRetriever

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
        
        self.dependancy_graph = DependancyGraph(
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
    
    def request(self, user_query:str):
        user_query = user_query.strip()
        self.dependancy_graph.user_query_to_requirements(
            user_query,
            schema_context=self.format_schema()
            )
        
        self.dependancy_graph.process_dependancy_graph(
                schema_context=self.format_schema()
        )
        
        return "Waiting for response from explainer..."
