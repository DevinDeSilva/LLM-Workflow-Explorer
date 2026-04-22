import json
from typing import Any, Dict, List

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
                 synthetic_question_loc:str,
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
            ttl_config,
            synthetic_question_loc
        )
        
    def format_schema(self) -> str: 
        return self.ontology_info.format_schema_prompt()
    
    def request(self, user_query:str):
        user_query = user_query.strip()
        application_context = (self.app_info.description or "").strip()
        
        try:
            self.dependancy_graph.user_query_to_requirements(
                user_query,
                schema_context=self.format_schema(),
                application_context=application_context,
                )
            
            return self.dependancy_graph.process_dependancy_graph(
                schema_context=self.format_schema(),
                application_context=application_context,
            )
        except ValueError as e:
            return {"error": str(e)}
        
    def request_to_report(self, data:Dict[str, Any]):
        def truncate_text(value: Any, limit: int = 1200) -> str:
            text = str(value or "").strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3].rstrip() + "..."

        def compact_value(value: Any, depth: int = 0) -> Any:
            if depth >= 3:
                return truncate_text(value, 400)
            if isinstance(value, dict):
                items = list(value.items())
                compacted = {
                    str(key): compact_value(item, depth + 1)
                    for key, item in items[:3]
                }
                if len(items) > 3:
                    compacted["_truncated_keys"] = len(items) - 3
                return compacted
            if isinstance(value, list):
                compacted = [compact_value(item, depth + 1) for item in value[:3]]
                if len(value) > 3:
                    compacted.append(f"... {len(value) - 3} more item(s)")
                return compacted
            if isinstance(value, str):
                return truncate_text(value, 400)
            return value

        def compact_json(value: Any) -> str:
            if not value:
                return ""
            return json.dumps(compact_value(value), indent=2, ensure_ascii=False)

        def flatten_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
            nodes: List[Dict[str, Any]] = []

            def visit(current: Dict[str, Any]) -> None:
                predecessors = current.get("predecessor_info", {})
                if isinstance(predecessors, dict):
                    for predecessor in predecessors.values():
                        if isinstance(predecessor, dict):
                            visit(predecessor)
                nodes.append(current)

            visit(node if isinstance(node, dict) else {})
            return nodes

        nodes = flatten_nodes(data)
        final_judge = data.get("judge", []) if isinstance(data, dict) else []
        final_feedback = ""
        if isinstance(final_judge, list) and final_judge:
            final_feedback = str(final_judge[-1].get("feedback", "")).strip()

        lines = [
            "### Overall Answer to the Question:",
            truncate_text(data.get("answer", "")),
            "",
            "### Original Question:",
            str(data.get("question", "")).strip(),
        ]

        if final_feedback:
            lines.extend(
                [
                    "",
                    "### Final Validation:",
                    truncate_text(final_feedback, 800),
                ]
            )

        if len(nodes) > 1:
            lines.extend(
                [
                    "",
                    "### Sub Questions Used:",
                ]
            )
            for node in nodes[:-1]:
                node_id = str(node.get("id", "")).strip()
                question = str(node.get("question", "")).strip()
                answer = truncate_text(node.get("answer", ""), 500)
                lines.append(f"- [{node_id}] {question}")
                if answer:
                    lines.append(f"  Answer: {answer}")

        lines.extend(
            [
                "",
                "### Grounded Reasoning Trace:",
            ]
        )

        for node in nodes:
            node_id = str(node.get("id", "")).strip()
            question = str(node.get("question", "")).strip()
            answer = truncate_text(node.get("answer", ""), 700)
            lines.append(f"Question [{node_id}]: {question}")
            if answer:
                lines.append(f"Answer: {answer}")

            plan = node.get("plan", [])
            if isinstance(plan, list) and plan:
                lines.append("Planned steps:")
                for planned_step in plan:
                    if not isinstance(planned_step, dict):
                        continue
                    step_id = str(planned_step.get("step_id", "")).strip()
                    sub_question = str(
                        planned_step.get("sub_question", "")
                    ).strip()
                    program_id = str(planned_step.get("program_id", "")).strip()
                    lines.append(
                        f"- {step_id or 'step'} | {sub_question} | program={program_id}"
                    )

            execution = node.get("execution", [])
            if isinstance(execution, list) and execution:
                lines.append("Executed steps and data:")
                for step in execution:
                    if not isinstance(step, dict):
                        continue
                    step_id = str(step.get("step_id", "")).strip() or "step"
                    sub_question = str(step.get("sub_question", "")).strip()
                    program_id = str(step.get("program_id", "")).strip()
                    step_answer = truncate_text(step.get("answer", ""), 500)
                    important_entities = step.get("important_entities", [])
                    if not isinstance(important_entities, list):
                        important_entities = [important_entities]
                    lines.append(
                        f"- {step_id} | {sub_question} | program={program_id}"
                    )
                    if step_answer:
                        lines.append(f"  Step answer: {step_answer}")
                    if important_entities:
                        lines.append(
                            "  Important entities: "
                            + ", ".join(map(str, important_entities[:8]))
                        )
                    results_preview = compact_json(step.get("results", {}))
                    if results_preview:
                        lines.append("  Data:")
                        lines.append(results_preview)

            grounding_preview = compact_json(node.get("grounding", {}))
            if grounding_preview:
                lines.append("Grounding:")
                lines.append(grounding_preview)

            judge = node.get("judge", [])
            if isinstance(judge, list) and judge:
                judge_feedback = str(judge[-1].get("feedback", "")).strip()
                if judge_feedback:
                    lines.append(
                        "Validation: " + truncate_text(judge_feedback, 500)
                    )

            lines.append("")

        return "\n".join(lines).strip()
