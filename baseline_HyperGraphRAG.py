from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import dycomutils as common_utils
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.HyperGraphRAGKG import HyperGraphKG, build_hypergraph_kg, write_hypergraph_kg
from src.config.experiment import FullContextExperimentConfig
from src.experiment.ground_truth import GT, GTInfo
from src.utils.utils import create_timestamp_id, load_config


REPO_ROOT = Path(__file__).resolve().parent
HYPERGRAPH_ROOT = REPO_ROOT / "baselines" / "HyperGraphRAG"
GRAPH_FIELD_SEP = "<SEP>"


def _get_evaluation_choices() -> list[str]:
    evaluations_dir = REPO_ROOT / "evaluations"
    return sorted(
        path.name
        for path in evaluations_dir.iterdir()
        if path.is_dir() and path.name != "test_questions"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HyperGraphRAG over this repo's KG.")
    parser.add_argument(
        "--evaluation",
        choices=_get_evaluation_choices(),
        default="chatbs-base",
        help="Evaluation folder under evaluations/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path. Defaults to evaluations/<evaluation>/config.hypergraphrag.yaml.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the KG-to-hypergraph JSON; do not import or run HyperGraphRAG.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of questions to answer.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of KG object segments to index.",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Delete the HyperGraphRAG working directory before injecting the KG.",
    )
    return parser.parse_args()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dataclass
class OpenAIChatUsageTracker:
    model: str
    base_url: Optional[str] = None
    api_key: str = "sk-"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        from openai import AsyncOpenAI

        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)
        kwargs.pop("response_format", None)
        stream = bool(kwargs.pop("stream", False))
        if stream:
            raise ValueError("Streaming HyperGraphRAG responses are not supported by this runner.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or self.api_key,
        )
        response = await client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        usage_dict = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        if not usage_dict["total_tokens"]:
            usage_dict["total_tokens"] = (
                usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]
            )
        self.calls.append(
            {
                "model": self.model,
                "usage": usage_dict,
                "estimated": False,
                "source": "provider",
            }
        )
        return content

    def reset(self) -> None:
        self.calls.clear()

    def token_usage(self) -> dict[str, Any]:
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        models: dict[str, dict[str, int]] = {}
        for call in self.calls:
            model = call["model"]
            usage = call["usage"]
            model_usage = models.setdefault(
                model,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
            for key in totals:
                value = int(usage.get(key, 0) or 0)
                totals[key] += value
                model_usage[key] += value
        return {
            **totals,
            "models": models,
            "calls": list(self.calls),
            "estimated": False,
            "source": "provider" if self.calls else "none",
        }


def _make_embedding_func(
    *,
    model: str,
    base_url: Optional[str],
    embedding_dim: int,
    max_token_size: int,
):
    if str(HYPERGRAPH_ROOT) not in sys.path:
        sys.path.insert(0, str(HYPERGRAPH_ROOT))
    from hypergraphrag.utils import EmbeddingFunc

    async def _embed(texts: list[str]) -> np.ndarray:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "sk-",
        )
        response = await client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )
        return np.array([item.embedding for item in response.data])

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=_embed,
    )


async def _ainject_hypergraph_kg(rag: Any, hypergraph_kg: HyperGraphKG) -> None:
    from hypergraphrag.utils import compute_mdhash_id

    full_docs: dict[str, dict[str, Any]] = {}
    text_chunks: dict[str, dict[str, Any]] = {}
    source_to_chunk_id: dict[str, str] = {}

    for index, chunk in enumerate(hypergraph_kg.chunks):
        content = chunk["content"].strip()
        doc_id = compute_mdhash_id(content, prefix="doc-")
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        source_to_chunk_id[chunk["source_id"]] = chunk_id
        full_docs[doc_id] = {"content": content}
        text_chunks[chunk_id] = {
            "tokens": 0,
            "content": content,
            "full_doc_id": doc_id,
            "chunk_order_index": index,
        }

    await rag.full_docs.upsert(full_docs)
    await rag.text_chunks.upsert(text_chunks)
    await rag.chunks_vdb.upsert(text_chunks)

    for entity in hypergraph_kg.entities:
        source_ids = [
            source_to_chunk_id.get(source_id, source_id)
            for source_id in str(entity["source_id"]).split(GRAPH_FIELD_SEP)
        ]
        node_data = {
            "role": "entity",
            "entity_type": entity.get("entity_type", "UNKNOWN"),
            "description": entity.get("description", ""),
            "source_id": GRAPH_FIELD_SEP.join(source_ids),
        }
        await rag.chunk_entity_relation_graph.upsert_node(
            entity["entity_name"],
            node_data=node_data,
        )

    for hyperedge in hypergraph_kg.hyperedges:
        source_id = source_to_chunk_id.get(hyperedge["source_id"], hyperedge["source_id"])
        hyperedge_name = hyperedge["hyperedge_name"]
        await rag.chunk_entity_relation_graph.upsert_node(
            hyperedge_name,
            node_data={
                "role": "hyperedge",
                "weight": float(hyperedge.get("weight", 1.0)),
                "source_id": source_id,
            },
        )
        for entity_name in hyperedge.get("entity_names", []):
            if not await rag.chunk_entity_relation_graph.has_node(entity_name):
                await rag.chunk_entity_relation_graph.upsert_node(
                    entity_name,
                    node_data={
                        "role": "entity",
                        "entity_type": "UNKNOWN",
                        "description": entity_name,
                        "source_id": source_id,
                    },
                )
            await rag.chunk_entity_relation_graph.upsert_edge(
                hyperedge_name,
                entity_name,
                edge_data={
                    "weight": float(hyperedge.get("weight", 1.0)),
                    "source_id": source_id,
                },
            )

    entity_vdb_rows = {
        compute_mdhash_id(entity["entity_name"], prefix="ent-"): {
            "content": entity["entity_name"] + "\n" + entity.get("description", ""),
            "entity_name": entity["entity_name"],
        }
        for entity in hypergraph_kg.entities
    }
    hyperedge_vdb_rows = {
        compute_mdhash_id(hyperedge["hyperedge_name"], prefix="rel-"): {
            "content": (
                hyperedge["hyperedge_name"]
                + "\n"
                + hyperedge.get("description", "")
                + "\n"
                + " | ".join(hyperedge.get("entity_names", []))
            ),
            "hyperedge_name": hyperedge["hyperedge_name"],
        }
        for hyperedge in hypergraph_kg.hyperedges
    }

    await rag.entities_vdb.upsert(entity_vdb_rows)
    await rag.hyperedges_vdb.upsert(hyperedge_vdb_rows)
    await rag._insert_done()


def _extract_csv_section(context: str, section: str) -> list[dict[str, str]]:
    pattern = rf"-----{re.escape(section)}-----\s*```csv\s*(.*?)```"
    match = re.search(pattern, context, flags=re.DOTALL)
    if not match:
        return []
    body = match.group(1).strip()
    if not body:
        return []
    rows = list(csv.DictReader(io.StringIO(body)))
    return [dict(row) for row in rows]


def _clean_entity_label(entity_name: str) -> str:
    return str(entity_name).strip().strip('"').title()


def _prediction_from_context(
    *,
    qinfo: GT,
    answer: str,
    context: str,
    token_usage: dict[str, Any],
    time_taken: float,
) -> dict[str, Any]:
    entity_rows = _extract_csv_section(context, "Entities")
    relationship_rows = _extract_csv_section(context, "Relationships")
    source_rows = _extract_csv_section(context, "Sources")

    relevant_entities = [
        {
            "id": row.get("entity", ""),
            "label": _clean_entity_label(row.get("entity", "")),
            "types": [row.get("type", "UNKNOWN")],
            "score": 0.0,
        }
        for row in entity_rows
        if row.get("entity")
    ]
    evidence = [
        {
            "subject_id": row.get("hyperedge", ""),
            "subject_label": row.get("hyperedge", ""),
            "predicate_id": "hyperedge",
            "predicate_label": "hyperedge",
            "object_id": row.get("related_entities", ""),
            "object_label": row.get("related_entities", ""),
            "object_is_literal": False,
            "direction": "hyperedge",
            "score": 0.0,
        }
        for row in relationship_rows
        if row.get("hyperedge")
    ]
    retrieved_docs = [
        {
            "text": row.get("content", ""),
            "score": 0.0,
        }
        for row in source_rows
        if row.get("content")
    ]
    citations = " ".join(
        f"<cite, id={index}>{entity['label']}</cite>"
        for index, entity in enumerate(relevant_entities[:5])
    )
    cited_answer = f"{answer.strip()} {citations}".strip() if citations else answer.strip()

    return {
        "answer": cited_answer,
        "relevant_entities": relevant_entities,
        "evidence": evidence,
        "retrieved_docs": retrieved_docs,
        "token_usage": token_usage,
        "question": qinfo.question,
        "id": qinfo.id,
        "time_taken": time_taken,
    }


def _load_experiment_config(config_path: Path) -> FullContextExperimentConfig:
    logging.info("Loading config: %s", config_path)
    raw_config = load_config(str(config_path))
    return FullContextExperimentConfig.model_validate(raw_config)


def _paths(
    config: FullContextExperimentConfig,
    object_config: dict[str, Any],
) -> tuple[Path, Path]:
    working_dir = Path(
        str(
            object_config.get("hypergraphrag_working_dir")
            or Path(config.explainer_config.save_answer_loc) / "index"
        )
    )
    custom_kg_path = working_dir / "chatbs_hypergraph_kg.json"
    return working_dir, custom_kg_path


def main() -> None:
    args = _parse_args()
    load_dotenv()
    os.environ.setdefault("OPENAI_API_KEY", "sk-")

    config_path = Path(
        args.config
        or REPO_ROOT / "evaluations" / args.evaluation / "config.hypergraphrag.yaml"
    )
    config = _load_experiment_config(config_path)
    object_config = dict(config.explainer_config.object_search_config)
    os.makedirs(os.path.dirname(config.explainer_config.log_file), exist_ok=True)
    logging.basicConfig(
        filename=config.explainer_config.log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    working_dir, custom_kg_path = _paths(config, object_config)
    hypergraph_kg = build_hypergraph_kg(
        kg_path=config.file_paths.execution_kg_loc,
        ontology_path=config.file_paths.ontology_path,
        metadata_path=config.file_paths.metadata_loc,
        max_literal_chars=_as_optional_int(object_config.get("max_literal_chars"), 800),
        max_objects=args.max_objects,
    )
    write_hypergraph_kg(hypergraph_kg, custom_kg_path)

    print(
        "Prepared HyperGraphRAG KG with "
        f"{len(hypergraph_kg.chunks)} chunks, "
        f"{len(hypergraph_kg.entities)} entities, "
        f"{len(hypergraph_kg.hyperedges)} hyperedges."
    )
    print(f"Custom KG: {custom_kg_path}")

    if args.prepare_only:
        return

    force_index = args.force_index or _as_bool(
        object_config.get("force_index_from_scratch"),
        False,
    )
    if force_index and working_dir.exists():
        shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

    if str(HYPERGRAPH_ROOT) not in sys.path:
        sys.path.insert(0, str(HYPERGRAPH_ROOT))
    from hypergraphrag import HyperGraphRAG
    from hypergraphrag.base import QueryParam
    from hypergraphrag.prompt import PROMPTS

    llm_config = dict(config.explainer_config.llm_config)
    embedding_config = dict(config.explainer_config.embedding_config)
    llm_tracker = OpenAIChatUsageTracker(
        model=str(llm_config.get("model", "openai/gpt-oss-20b")),
        base_url=llm_config.get("base_url"),
        temperature=_as_float(llm_config.get("temperature"), 0.0),
        max_tokens=_as_optional_int(llm_config.get("max_tokens"), None),
    )
    embedding_func = _make_embedding_func(
        model=str(embedding_config.get("model", "text-embedding-bge-large-en-v1.5")),
        base_url=embedding_config.get("base_url", llm_config.get("base_url")),
        embedding_dim=_as_int(object_config.get("embedding_dim"), 1024),
        max_token_size=_as_int(object_config.get("embedding_max_token_size"), 8192),
    )

    rag = HyperGraphRAG(
        working_dir=str(working_dir),
        embedding_func=embedding_func,
        embedding_batch_num=_as_int(object_config.get("embedding_batch_num"), 16),
        embedding_func_max_async=_as_int(object_config.get("embedding_func_max_async"), 4),
        llm_model_func=llm_tracker.complete,
        llm_model_name=llm_tracker.model,
        llm_model_max_async=_as_int(object_config.get("llm_model_max_async"), 4),
        llm_model_max_token_size=_as_int(
            object_config.get("llm_model_max_token_size"),
            32768,
        ),
        enable_llm_cache=_as_bool(object_config.get("enable_llm_cache"), False),
        chunk_token_size=_as_int(object_config.get("chunk_token_size"), 1200),
        chunk_overlap_token_size=_as_int(
            object_config.get("chunk_overlap_token_size"),
            100,
        ),
    )
    _run_async(_ainject_hypergraph_kg(rag, hypergraph_kg))

    ground_truth = GTInfo(config.gt.save_loc)
    questions = ground_truth.gt_info
    if args.limit_questions is not None and args.limit_questions > 0:
        questions = questions[: args.limit_questions]

    timestamp_exp = create_timestamp_id("exp_")
    run_dir = Path(config.explainer_config.save_answer_loc) / timestamp_exp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "RESULTS.jsonl"

    query_param = QueryParam(
        mode="hybrid",
        only_need_context=True,
        response_type=str(object_config.get("response_type", "Single Paragraph")),
        top_k=_as_int(object_config.get("top_k"), 20),
        max_token_for_text_unit=_as_int(
            object_config.get("max_token_for_text_unit"),
            4000,
        ),
        max_token_for_global_context=_as_int(
            object_config.get("max_token_for_global_context"),
            4000,
        ),
        max_token_for_local_context=_as_int(
            object_config.get("max_token_for_local_context"),
            4000,
        ),
    )

    for qinfo in tqdm(questions):
        start_time = time.perf_counter()
        llm_tracker.reset()
        context = rag.query(qinfo.question, query_param)
        if context == PROMPTS["fail_response"]:
            answer = context
        else:
            system_prompt = PROMPTS["rag_response"].format(
                context_data=context,
                response_type=query_param.response_type,
            )
            answer = _run_async(
                llm_tracker.complete(
                    qinfo.question,
                    system_prompt=system_prompt,
                )
            )
        end_time = time.perf_counter()

        pred = _prediction_from_context(
            qinfo=qinfo,
            answer=answer,
            context=context,
            token_usage=llm_tracker.token_usage(),
            time_taken=end_time - start_time,
        )
        common_utils.serialization.save_jsonl_append(str(results_path), pred)

    run_config = config.model_dump()
    run_config["hypergraphrag"] = {
        "working_dir": str(working_dir),
        "custom_kg_path": str(custom_kg_path),
        "chunks": len(hypergraph_kg.chunks),
        "entities": len(hypergraph_kg.entities),
        "hyperedges": len(hypergraph_kg.hyperedges),
        "force_index_from_scratch": force_index,
    }
    common_utils.serialization.save_json(
        run_config,
        str(run_dir / "config.json"),
    )
    print(f"Saved HyperGraphRAG predictions to {results_path}")


if __name__ == "__main__":
    main()

