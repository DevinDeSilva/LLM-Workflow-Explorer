from src.llm import LLM
from src.embeddings import Embeddings
from src.vector_db import VectorDB
from src.explainer.dependancy_graph import DependancyGraphCreation
from src.config.experiment import ExplainerConfig, TTLConfig

class Explainer:
    def __init__(self, 
                 kg_loc:str,
                 config:ExplainerConfig,
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
        
        self.dependancy_graph = DependancyGraphCreation(
            kg_loc,
            self.llm,
            self.embedding,
            self.db,
            self.config.object_search_config,
            ttl_config
        )
    
    def request(self, user_query:str):
        user_query = user_query.strip()
        self.dependancy_graph.user_query_to_requirements(user_query)