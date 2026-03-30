from src.config.object_search import ObjectSearchConfig
from src.llm import LLM
from src.embeddings import Embeddings
from src.vector_db import VectorDB
from src.explainer.dependancy_graph import DependancyGraphCreation
from src.config.experiment import ExplainerConfig, TTLConfig

import dycomutils as common_utils

class Explainer:
    def __init__(self, 
                 kg_loc:str,
                 schema_loc:str,
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
        
        self.schema = common_utils.serialization.load_json(
            schema_loc
        )
        
        self.dependancy_graph = DependancyGraphCreation(
            kg_loc,
            self.llm,
            self.embedding,
            self.db,
            ObjectSearchConfig(**self.config.object_search_config),
            ttl_config
        )
        
    def format_schema(self) -> str: 
        return ""
    
    def request(self, user_query:str):
        user_query = user_query.strip()
        info = self.dependancy_graph.user_query_to_requirements(
            user_query,
            schema_context=self.format_schema()
            )
        return info