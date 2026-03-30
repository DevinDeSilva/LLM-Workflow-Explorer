import json
import re
from typing import Any, Dict, List, Optional

import dspy

from src.llm import LLM
from src.utils.graph_manager import GraphManager
from src.config.experiment import TTLConfig

class DependancyGraphCreation:
    def __init__(  
        self, 
        llm_type: str, 
        model: str, 
        llm_config: Dict[str, Any],
        graph_loc:str,
        ttl_config:TTLConfig
        
    ) -> None:
        self.llm = LLM(
            llm_type,
            "dspy",
            model=model,
            **llm_config,
        )
        
        self.graph_manager = GraphManager(
            graph_file=graph_loc,
            config=ttl_config
        )
        
        object_db = 

    def user_query_to_requirements(
        self, query: str, schema_context: str = ""
    ) -> Dict[str, Any]:
        pass
    
    def ambiguity_removal(self, query: str, schema_context: str = "") -> Dict[str, Any]:
        pass