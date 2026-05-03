from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import dycomutils as common_utils
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.HippoRAGKG import (
    KGDocument,
    build_kg_documents,
    documents_by_content,
    write_hipporag_corpus,
    write_hipporag_openie,
)
from src.config.experiment import FullContextExperimentConfig
from src.experiment.ground_truth import GT, GTInfo
from src.utils.utils import create_timestamp_id, load_config


REPO_ROOT = Path(__file__).resolve().parent
HIPPO_ROOT = REPO_ROOT / "baselines" / "HippoRAG"


def _get_evaluation_choices() -> list[str]:
    evaluations_dir = REPO_ROOT / "evaluations"
    return sorted(
        path.name
        for path in evaluations_dir.iterdir()
        if path.is_dir() and path.name != "test_questions"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HippoRAG over this repo's KG.")
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
        help="Config path. Defaults to evaluations/<evaluation>/config.hipporag.yaml.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the KG corpus and injected OpenIE file; do not import or run HippoRAG.",
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
        help="Rebuild HippoRAG graph data from the KG documents.",
    )
    parser.add_argument(
        "--force-openie",
        action="store_true",
        help="Ignore injected KG OpenIE and let HippoRAG run OpenIE with the LLM.",
    )
    parser.add_argument(
        "--no-kg-openie-injection",
        action="store_true",
        help="Do not pre-populate HippoRAG's OpenIE file from the KG.",
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


def _as_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def _normalize_usage(metadata: dict[str, Any], model: str) -> dict[str, Any]:
    prompt_tokens = int(metadata.get("prompt_tokens", 0) or 0)
    completion_tokens = int(metadata.get("completion_tokens", 0) or 0)
    total_tokens = int(metadata.get("total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    if not total_tokens:
        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models": {},
            "calls": [],
            "estimated": False,
            "source": "none",
        }

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    return {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "models": {model: usage},
        "calls": [
            {
                "model": model,
                "usage": usage,
                "estimated": False,
                "source": "provider",
                "cache_hit": bool(metadata.get("cache_hit", False)),
            }
        ],
        "estimated": False,
        "source": "provider",
    }


def _attach_entity_citations(answer: str, entities: list[dict[str, Any]]) -> str:
    cleaned_answer = answer.strip()
    if not entities:
        return cleaned_answer
    citations = " ".join(
        f"<cite, id={index}>{entity['label']}</cite>"
        for index, entity in enumerate(entities)
    )
    if not cleaned_answer:
        return citations
    return f"{cleaned_answer} {citations}"


def _scores(solution: Any) -> list[float]:
    doc_scores = getattr(solution, "doc_scores", None)
    if doc_scores is None:
        return []
    if hasattr(doc_scores, "tolist"):
        return [float(score) for score in doc_scores.tolist()]
    return [float(score) for score in doc_scores]


def _prediction_from_solution(
    *,
    qinfo: GT,
    solution: Any,
    metadata: dict[str, Any],
    model: str,
    document_lookup: dict[str, KGDocument],
    time_taken: float,
) -> dict[str, Any]:
    relevant_entities: list[dict[str, Any]] = []
    retrieved_docs: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    seen_evidence: set[tuple[str, str, Optional[str], str]] = set()

    scores = _scores(solution)
    for index, doc_text in enumerate(getattr(solution, "docs", []) or []):
        score = scores[index] if index < len(scores) else 0.0
        kg_doc = document_lookup.get(doc_text)
        if kg_doc is None:
            label = doc_text.splitlines()[0].strip() if doc_text else "Retrieved document"
            entity = {
                "id": label,
                "label": label,
                "types": [],
                "score": score,
            }
        else:
            entity = {
                "id": kg_doc.id,
                "label": kg_doc.label,
                "types": kg_doc.types,
                "score": score,
            }
            for triple in kg_doc.evidence:
                key = (
                    triple.subject_id,
                    triple.predicate_id,
                    triple.object_id,
                    triple.object_label,
                )
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                evidence_row = triple.to_dict()
                evidence_row["score"] = score
                evidence.append(evidence_row)

        if entity["id"] not in seen_entities:
            seen_entities.add(entity["id"])
            relevant_entities.append(entity)
        retrieved_docs.append({"text": doc_text, "score": score})

    answer = getattr(solution, "answer", "") or ""
    return {
        "answer": _attach_entity_citations(answer, relevant_entities),
        "relevant_entities": relevant_entities,
        "evidence": evidence,
        "retrieved_docs": retrieved_docs,
        "token_usage": _normalize_usage(metadata, model),
        "question": qinfo.question,
        "id": qinfo.id,
        "time_taken": time_taken,
    }


def _load_config(config_path: Path) -> FullContextExperimentConfig:
    logging.info("Loading config: %s", config_path)
    raw_config = load_config(str(config_path))
    return FullContextExperimentConfig.model_validate(raw_config)


def _prepare_kg_inputs(
    config: FullContextExperimentConfig,
    object_config: dict[str, Any],
    *,
    max_objects: Optional[int] = None,
) -> list[KGDocument]:
    return build_kg_documents(
        kg_path=config.file_paths.execution_kg_loc,
        ontology_path=config.file_paths.ontology_path,
        metadata_path=config.file_paths.metadata_loc,
        max_literal_chars=_as_optional_int(
            object_config.get("max_literal_chars"),
            800,
        ),
        max_objects=max_objects,
    )


def _hipporag_paths(
    config: FullContextExperimentConfig,
    object_config: dict[str, Any],
    llm_model: str,
) -> tuple[Path, Path, Path]:
    save_dir = Path(
        str(
            object_config.get("hipporag_save_dir")
            or Path(config.explainer_config.save_answer_loc) / "index"
        )
    )
    corpus_path = save_dir / "chatbs_kg_corpus.json"
    openie_path = save_dir / f"openie_results_ner_{llm_model.replace('/', '_')}.json"
    return save_dir, corpus_path, openie_path


def _write_injected_inputs(
    documents: list[KGDocument],
    corpus_path: Path,
    openie_path: Path,
) -> None:
    write_hipporag_corpus(documents, corpus_path)
    write_hipporag_openie(documents, openie_path)


def _build_hipporag_config(
    *,
    config: FullContextExperimentConfig,
    object_config: dict[str, Any],
    save_dir: Path,
    corpus_len: int,
    force_index: bool,
    force_openie: bool,
):
    if str(HIPPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(HIPPO_ROOT / "src"))
    from hipporag.utils.config_utils import BaseConfig

    llm_config = dict(config.explainer_config.llm_config)
    embedding_config = dict(config.explainer_config.embedding_config)
    llm_model = str(llm_config.get("model", "gpt-4o-mini"))
    llm_base_url = llm_config.get("base_url")
    embedding_model = str(
        embedding_config.get("model", "text-embedding-bge-large-en-v1.5")
    )
    embedding_base_url = embedding_config.get("base_url", llm_base_url)

    return BaseConfig(
        save_dir=str(save_dir),
        llm_base_url=str(llm_base_url) if llm_base_url else None,
        llm_name=llm_model,
        embedding_base_url=str(embedding_base_url) if embedding_base_url else None,
        embedding_model_name=embedding_model,
        force_index_from_scratch=force_index,
        force_openie_from_scratch=force_openie,
        rerank_dspy_file_path=str(
            HIPPO_ROOT
            / "src"
            / "hipporag"
            / "prompts"
            / "dspy_prompts"
            / "filter_llama3.3-70B-Instruct.json"
        ),
        retrieval_top_k=_as_int(object_config.get("retrieval_top_k"), 50),
        linking_top_k=_as_int(object_config.get("linking_top_k"), 5),
        max_qa_steps=_as_int(object_config.get("max_qa_steps"), 1),
        qa_top_k=_as_int(object_config.get("qa_top_k"), 5),
        graph_type=str(
            object_config.get(
                "graph_type",
                "facts_and_sim_passage_node_unidirectional",
            )
        ),
        embedding_batch_size=_as_int(object_config.get("embedding_batch_size"), 8),
        max_new_tokens=_as_optional_int(object_config.get("max_new_tokens"), 800),
        corpus_len=corpus_len,
        openie_mode=str(object_config.get("openie_mode", "online")),
        dataset=str(object_config.get("prompt_dataset", "musique")),
    )


def main() -> None:
    args = _parse_args()
    load_dotenv()

    config_path = Path(
        args.config
        or REPO_ROOT / "evaluations" / args.evaluation / "config.hipporag.yaml"
    )
    config = _load_config(config_path)
    object_config = dict(config.explainer_config.object_search_config)
    os.makedirs(os.path.dirname(config.explainer_config.log_file), exist_ok=True)
    logging.basicConfig(
        filename=config.explainer_config.log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    llm_model = str(config.explainer_config.llm_config.get("model", "gpt-4o-mini"))
    save_dir, corpus_path, openie_path = _hipporag_paths(
        config,
        object_config,
        llm_model,
    )
    documents = _prepare_kg_inputs(
        config,
        object_config,
        max_objects=args.max_objects,
    )

    inject_kg_openie = not args.no_kg_openie_injection and _as_bool(
        object_config.get("inject_kg_openie"),
        True,
    )
    if inject_kg_openie:
        _write_injected_inputs(documents, corpus_path, openie_path)
    else:
        write_hipporag_corpus(documents, corpus_path)

    print(f"Prepared {len(documents)} KG object documents for HippoRAG.")
    print(f"Corpus: {corpus_path}")
    if inject_kg_openie:
        print(f"Injected OpenIE: {openie_path}")

    if args.prepare_only:
        return

    if str(HIPPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(HIPPO_ROOT / "src"))
    from hipporag import HippoRAG

    force_index = args.force_index or _as_bool(
        object_config.get("force_index_from_scratch"),
        False,
    )
    force_openie = args.force_openie or _as_bool(
        object_config.get("force_openie_from_scratch"),
        False,
    )
    if inject_kg_openie and not args.force_openie:
        force_openie = False

    hippo_config = _build_hipporag_config(
        config=config,
        object_config=object_config,
        save_dir=save_dir,
        corpus_len=len(documents),
        force_index=force_index,
        force_openie=force_openie,
    )
    hipporag = HippoRAG(global_config=hippo_config)
    hipporag.index([document.content for document in documents])

    ground_truth = GTInfo(config.gt.save_loc)
    questions = ground_truth.gt_info
    if args.limit_questions is not None and args.limit_questions > 0:
        questions = questions[: args.limit_questions]

    timestamp_exp = create_timestamp_id("exp_")
    run_dir = Path(config.explainer_config.save_answer_loc) / timestamp_exp
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "RESULTS.jsonl"
    document_lookup = documents_by_content(documents)

    for qinfo in tqdm(questions):
        start_time = time.perf_counter()
        result = hipporag.rag_qa(queries=[qinfo.question])
        end_time = time.perf_counter()
        solutions, _messages, metadata = result[:3]
        pred = _prediction_from_solution(
            qinfo=qinfo,
            solution=solutions[0],
            metadata=metadata[0] if metadata else {},
            model=llm_model,
            document_lookup=document_lookup,
            time_taken=end_time - start_time,
        )
        common_utils.serialization.save_jsonl_append(str(results_path), pred)

    run_config = config.model_dump()
    run_config["hipporag"] = {
        "kg_object_documents": len(documents),
        "hipporag_save_dir": str(save_dir),
        "corpus_path": str(corpus_path),
        "openie_path": str(openie_path) if inject_kg_openie else None,
        "kg_openie_injected": inject_kg_openie,
        "force_index_from_scratch": force_index,
        "force_openie_from_scratch": force_openie,
    }
    common_utils.serialization.save_json(
        run_config,
        str(run_dir / "config.json"),
    )
    print(f"Saved HippoRAG predictions to {results_path}")


if __name__ == "__main__":
    main()
