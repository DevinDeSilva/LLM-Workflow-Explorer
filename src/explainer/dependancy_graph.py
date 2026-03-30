import json
import re
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel
from icecream import ic

from src.config.object_search import ObjectSearchConfig
from src.embeddings.base import BaseEmbeddings
from src.explainer.object_search import ObjectSearch
from src.llm.base import BaseLLM
from src.utils.graph_manager import GraphManager
from src.config.experiment import TTLConfig
from src.vector_db.base import BaseVectorDB

class QuestionNode(BaseModel):
    pass

class DependancyGraphCreation:
    def __init__(  
        self, 
        graph_loc:str,
        llm: BaseLLM,
        embedder:BaseEmbeddings,
        vector_db:BaseVectorDB,
        object_search_config:ObjectSearchConfig,
        ttl_config:TTLConfig
    ) -> None:
        
        self.graph_manager = GraphManager(
            graph_file=graph_loc,
            config=ttl_config
        )
        
        self.llm = llm
        self.embedder = embedder
        self.vector_db = vector_db
        
        self.object_db = ObjectSearch(
            self.graph_manager,
            self.embedder,
            self.vector_db,
            object_search_config
        )

    def user_query_to_requirements(
        self, query: str, schema_context: str = ""
    ) -> Dict[str, Any]:
        ic(query)
        ic(schema_context)
        return {}
    
    def ambiguity_removal(self, query: str, schema_context: str = "") -> Dict[str, Any]:
        pass