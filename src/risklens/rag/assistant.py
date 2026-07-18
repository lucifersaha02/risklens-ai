"""Guarded, citation-backed analyst evidence assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from risklens.config import CONFIG_DIR
from risklens.rag.knowledge_base import (
    LocalKnowledgeIndex,
    load_knowledge_index,
    validate_knowledge_query,
)

ASSISTANT_CONFIG_PATH = CONFIG_DIR / "assistant.yaml"

DECISION_REQUEST_PATTERNS = (
    re.compile(r"\b(approve|decline|deny|reject)\b.{0,60}\b(loan|application|applicant)\b", re.I),
    re.compile(r"\bshould\b.{0,40}\b(approve|decline|deny|reject|lend)\b", re.I),
    re.compile(r"\b(recommend|make)\b.{0,40}\b(credit|lending|loan)\s+decision\b", re.I),
    re.compile(r"\b(adverse action|denial)\s+(notice|reason)\b", re.I),
)


@dataclass(frozen=True)
class AssistantResponse:
    """Structured response whose claims are traceable to retrieved evidence."""

    question: str
    answer_type: str
    summary: str
    evidence: list[dict[str, Any]]
    citations: list[str]
    disclosures: list[str]
    human_review_required: bool
    generated_by_llm: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question": self.question,
            "answer_type": self.answer_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "citations": self.citations,
            "disclosures": self.disclosures,
            "human_review_required": self.human_review_required,
            "generated_by_llm": self.generated_by_llm,
        }


def load_assistant_config(path: Path = ASSISTANT_CONFIG_PATH) -> dict[str, Any]:
    """Load the version-controlled assistant policy."""
    if not path.exists():
        raise FileNotFoundError(f"Assistant configuration not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "assistant" not in config or "guardrails" not in config:
        raise ValueError("Assistant configuration is incomplete")
    return config


def validate_assistant_question(question: str) -> str:
    """Apply retrieval and credit-decision guardrails to an analyst question."""
    normalized = validate_knowledge_query(question)
    if any(pattern.search(normalized) for pattern in DECISION_REQUEST_PATTERNS):
        raise ValueError(
            "Individual or autonomous credit-decision advice is prohibited; "
            "ask about documented model, policy, governance, or monitoring evidence"
        )
    return normalized


def _unique_citations(results: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for result in results:
        citation = str(result["citation"])
        if citation not in citations:
            citations.append(citation)
    return citations


def _expand_retrieval_query(question: str) -> str:
    """Add controlled domain vocabulary for common analyst shorthand."""
    additions: list[str] = []
    lowered = question.lower()
    if "result" in lowered or "performance" in lowered:
        additions.append("performance metrics ROC-AUC PR-AUC Brier recall precision approval")
    if "fair" in lowered or "subgroup" in lowered:
        additions.append("responsible AI recall FPR selection rate calibration gap")
    if "drift" in lowered or "monitor" in lowered:
        additions.append("PSI severity feature alert unlabeled population")
    return " ".join([question, *additions])


def answer_governance_question(
    question: str,
    index: LocalKnowledgeIndex | None = None,
    config_path: Path = ASSISTANT_CONFIG_PATH,
) -> AssistantResponse:
    """Create a deterministic evidence briefing without autonomous decisions."""
    config = load_assistant_config(config_path)
    assistant_config = config["assistant"]
    question = validate_assistant_question(question)
    if len(question) > int(assistant_config["maximum_question_characters"]):
        raise ValueError("Assistant question exceeds the configured maximum length")

    index = index or load_knowledge_index()
    results = index.search(
        _expand_retrieval_query(question),
        top_k=int(assistant_config["default_top_k"]),
        minimum_score=float(assistant_config["minimum_score"]),
    )
    if not results:
        return AssistantResponse(
            question=question,
            answer_type="insufficient_evidence",
            summary="The trusted RiskLens knowledge base does not contain sufficient evidence.",
            evidence=[],
            citations=[],
            disclosures=list(config["required_disclosures"]),
            human_review_required=True,
        )

    maximum_characters = int(assistant_config["maximum_evidence_characters"])
    evidence = [
        {
            "citation": result["citation"],
            "relevance_score": result["score"],
            "excerpt": str(result["text"])[:maximum_characters],
        }
        for result in results
    ]
    citations = _unique_citations(results)
    summary = (
        f"Retrieved {len(evidence)} relevant evidence passages from "
        f"{len(citations)} cited documentation sections. Review the excerpts below; "
        "no uncited interpretation or individual credit recommendation was generated."
    )
    return AssistantResponse(
        question=question,
        answer_type="grounded_evidence_briefing",
        summary=summary,
        evidence=evidence,
        citations=citations,
        disclosures=list(config["required_disclosures"]),
        human_review_required=True,
    )
