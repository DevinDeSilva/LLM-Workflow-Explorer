import json
import re
from typing import Any, Dict, List, Optional

import dspy

from src.llm import LLM


class QueryToRequirementsSignature(dspy.Signature):
    """Convert a user question into a dependency graph of extraction requirements."""

    query = dspy.InputField(desc="Natural-language QA question from the user.")
    schema_context = dspy.InputField(
        desc="Optional knowledge graph schema or ontology summary."
    )
    plan_json = dspy.OutputField(
        desc=(
            "Return strict JSON with keys: answer_type, entities, requirements, edges, "
            "execution_order, assumptions. Each requirement must have id, label, "
            "description, requirement_type, and depends_on."
        )
    )


class QueryToRequirementsModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(QueryToRequirementsSignature)

    def forward(self, query: str, schema_context: str = ""):
        return self.predict(query=query, schema_context=schema_context)


class DependancyGraphCreation:
    def __init__(  
        self, llm_type: str, model: str, llm_config: Dict[str, Any]
    ) -> None:
        self.llm = LLM(
            llm_type,
            "dspy",
            model=model,
            **llm_config,
        )
        self.module = QueryToRequirementsModule()

    def user_query_to_requirements(
        self, query: str, schema_context: str = ""
    ) -> Dict[str, Any]:
        pass
    
    def ambiguity_removal(self, query: str, schema_context: str = "") -> Dict[str, Any]:
        pass