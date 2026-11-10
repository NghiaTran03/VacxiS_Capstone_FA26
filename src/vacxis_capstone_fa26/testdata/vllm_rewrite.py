"""Qwen2.5-vLLM backed query rewriter and reference answer generator.

This module is the GPU path of the query-driven dataset generator. It is
designed to run on Colab/Kaggle where a CUDA-enabled vLLM and the
Qwen2.5-7B-Instruct weights are available, while remaining importable
everywhere: vLLM is *never* imported at module load time, so the rest of the
project stays usable on CPU-only machines without vLLM installed.

Two capabilities are exposed:

* :class:`QwenRewriter` -- rewrites raw (out-of-domain) user queries into
  vaccine-domain questions that name a real target vaccine from the corpus.
* :class:`QwenAnswerGenerator` -- fills ``REFERENCE_ANSWER`` and
  ``CLAIM_EVIDENCE_MAP`` from the grounded ``RELEVANT_CHUNKS`` using the same
  JSON contract as the OpenRouter generator in :mod:`answer_generator`.

Both share a single :class:`VLLMClient` so the model is loaded exactly once
per run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .answer_generator import REFERENCE_ANSWER_FORMAT, SYSTEM_PROMPT, UNANSWERABLE_FALLBACK_ANSWER
from .taxonomy import QUERY_TYPE_DEFS

# Lazy, optional vLLM import. vLLM (and torch) are only present in the GPU
# environment (Colab/Kaggle). Keeping this import guarded means
# ``import vacxis_capstone_fa26.testdata`` stays cheap and offline-safe.
try:  # pragma: no cover - exercised only where vLLM is installed
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None


# Canonical Qwen instruct model. Overridable via --vllm-model.
DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The ten real query types, pulled from the taxonomy so the model must pick
# from exactly the values the rest of the pipeline understands.
QUERY_TYPE_VALUES: tuple[str, ...] = tuple(t["type"] for t in QUERY_TYPE_DEFS)

ANSWERABILITY_VALUES: tuple[str, ...] = (
    "ANSWERABLE",
    "PARTIALLY_ANSWERABLE",
    "UNANSWERABLE",
)


class VLLMError(RuntimeError):
    """Raised when vLLM is requested but unavailable or misconfigured."""


class Rewriter(Protocol):
    """Structural type: maps source queries to grounded vaccine queries."""

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one result dict per input query, preserving order.

        Each result carries at least: ``id``, ``vaccine``,
        ``rewritten_query``, ``query_type``, ``answerability``.
        """
        ...


# JSON-ish block the model may wrap structured output in.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a model JSON payload, tolerating ``` fences."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} block, if any.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return data if isinstance(data, dict) else None


def _normalize_result(raw: dict[str, Any], fallback_id: Any) -> dict[str, Any]:
    """Coerce a raw model payload into the rewriter result contract."""
    query_type = str(raw.get("query_type", "")).strip().upper()
    if query_type not in QUERY_TYPE_VALUES:
        query_type = "PRECAUTION"  # safe, broad category for unparseable output

    answerability = str(raw.get("answerability", "ANSWERABLE")).strip().upper()
    if answerability not in ANSWERABILITY_VALUES:
        answerability = "ANSWERABLE"

    return {
        "id": raw.get("id", fallback_id),
        "vaccine": str(raw.get("vaccine", "")).strip(),
        "rewritten_query": str(raw.get("rewritten_query", "")).strip(),
        "query_type": query_type,
        "answerability": answerability,
        "rationale": str(raw.get("rationale", "")).strip(),
    }


@dataclass
class VLLMClient:
    """Holds the lazily-constructed vLLM engine and sampling settings.

    Constructed once and shared between the rewriter and the answer
    generator so the (large) model is loaded a single time.
    """

    model: str = DEFAULT_VLLM_MODEL
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    dtype: str = "float16"
    seed: int | None = None
    llm: Any = None

    def ensure_loaded(self) -> None:
        """Load the model on first use; raise a clear error if no vLLM."""
        if self.llm is not None:
            return
        if LLM is None:
            raise VLLMError(
                "vLLM is not installed. Install it on a GPU host "
                "(Colab/Kaggle) with: pip install vllm, or pass "
                "--no-vllm to use the offline rule-based path."
            )
        self.llm = LLM(
            model=self.model,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            trust_remote_code=True,
        )

    def sample(self, **overrides: Any) -> Any:
        if SamplingParams is None:
            raise VLLMError("vLLM SamplingParams unavailable; is vllm installed?")
        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        kwargs.update(overrides)
        return SamplingParams(**kwargs)


_REWRITE_SYSTEM = """You are a medical query adapter. The user gives a Vietnamese
medical question that is NOT necessarily about vaccines. Rewrite it into a
natural Vietnamese question ABOUT VACCINES that preserves the user's original
clinical intent and any named symptoms/conditions. The rewritten question MUST
name exactly one real vaccine and target one of the listed query types.
Respond with a single JSON object and nothing else."""

_REWRITE_USER_TMPL = """Available vaccines:
{vaccines}

Allowed query_type values: {qtypes}
Allowed answerability values: ANSWERABLE, PARTIALLY_ANSWERABLE, UNANSWERABLE

Source query (id={qid}):
"""


class QwenRewriter:
    """Rewrites out-of-domain queries into grounded vaccine-domain queries.

    Grounding is *post-hoc*: the model proposes a ``vaccine`` name and the
    caller (:func:`generator.build_queries_from_source`) resolves it against
    ``index_by_vaccine`` so every record points at a real chunk. An
    unknown/empty vaccine degrades gracefully to a random corpus vaccine.
    """

    def __init__(
        self,
        client: VLLMClient,
        *,
        vaccine_names: list[str],
        chat_template: str = "chat",
    ) -> None:
        self.client = client
        self.vaccine_names = list(vaccine_names)
        self.chat_template = chat_template

    def rewrite(self, source_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not source_queries:
            return []
        self.client.ensure_loaded()

        vaccines = ", ".join(self.vaccine_names) if self.vaccine_names else "(none)"
        prompts: list[dict[str, Any]] = []
        for sq in source_queries:
            user = _REWRITE_USER_TMPL.format(
                vaccines=vaccines,
                qtypes=", ".join(QUERY_TYPE_VALUES),
                qid=sq.get("id"),
            )
            user += f"{sq.get('query', '').strip()}\n\n"
            user += (
                "Return JSON: {\"id\": <int>, \"vaccine\": \"<name from the "
                "available list>\", \"rewritten_query\": \"<Vietnamese "
                "vaccine question>\", \"query_type\": \"<allowed value>\", "
                "\"answerability\": \"<allowed value>\", \"rationale\": "
                "\"<short reason>\"}"
            )
            prompts.append(
                {
                    "prompt": [
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "chat_template": self.chat_template,
                }
            )

        sampling = self.client.sample()
        outputs = self.client.llm.generate(prompts, sampling)

        results: list[dict[str, Any]] = []
        for sq, out in zip(source_queries, outputs):
            raw_text = out.outputs[0].text if out and out.outputs else ""
            parsed = _extract_json(raw_text)
            results.append(_normalize_result(parsed or {}, sq.get("id")))
        return results


# Answers reuse the exact JSON schema and system prompt of the OpenRouter
# generator so the two paths produce byte-compatible record fields.
_JSON_SCHEMA_HINT = json.dumps(REFERENCE_ANSWER_FORMAT["json_schema"]["schema"])


class QwenAnswerGenerator:
    """Offline-capable, vLLM-backed reference answer generator.

    Implements the :class:`AnswerGenerator` protocol
    (``generate_answers(queries) -> queries``). For UNANSWERABLE queries or
    queries without a score>=3 chunk it mirrors
    :class:`ExtractiveAnswerGenerator` (fallback answer, empty claim map).
    """

    def __init__(self, client: VLLMClient) -> None:
        self.client = client

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for q in queries:
            has_evidence = any(
                c.get("RELEVANCE_SCORE", 0) >= 3 for c in q.get("RELEVANT_CHUNKS", [])
            )
            if q.get("ANSWERABILITY") == "UNANSWERABLE" or not has_evidence:
                q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
                q["CLAIM_EVIDENCE_MAP"] = []
            else:
                self._generate_single(q)
        return queries

    def _generate_single(self, q: dict[str, Any]) -> None:
        from .answer_generator import UNANSWERABLE_FALLBACK_ANSWER

        self.client.ensure_loaded()
        relevant = [
            c
            for c in q.get("RELEVANT_CHUNKS", [])
            if c.get("RELEVANCE_SCORE", 0) >= 3
        ]
        chunks_context = "\n\n---\n\n".join(
            f"CHUNK_ID: {c.get('CHUNK_ID')}\nNỘI DUNG:\n{c.get('TEXT')}"
            for c in relevant
        )
        user = (
            f"ANSWERABILITY: {q.get('ANSWERABILITY')}\nCÂU HỎI: {q.get('QUERY')}\n\n"
            f"CÁC ĐOẠN CHUNKS CUNG CẤP:\n{chunks_context}\n\n"
            "Trả JSON theo schema:\n"
            f"{_JSON_SCHEMA_HINT}"
        )
        prompt = {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "chat_template": "chat",
        }
        sampling = self.client.sample(max_tokens=1024)
        outputs = self.client.llm.generate([prompt], sampling)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        data = _extract_json(text)
        if not data or "reference_answer" not in data:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return
        q["REFERENCE_ANSWER"] = str(data.get("reference_answer", "")).strip()
        q["CLAIM_EVIDENCE_MAP"] = data.get("claim_evidence_map", []) or []
