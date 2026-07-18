"""Trusted-source chunking, retrieval, citations, and RAG evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from risklens.config import CONFIG_DIR, METRICS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, REPORT_DIR

RAG_CONFIG_PATH = CONFIG_DIR / "rag.yaml"
RAG_EVALUATION_CONFIG_PATH = CONFIG_DIR / "rag_evaluation.yaml"
RAG_INDEX_PATH = PROCESSED_DATA_DIR / "rag_knowledge_index.joblib"
RAG_MANIFEST_PATH = PROCESSED_DATA_DIR / "rag_knowledge_manifest.json"
RAG_EVALUATION_METRICS_PATH = METRICS_DIR / "rag_retrieval_evaluation.json"
RAG_EVALUATION_REPORT_PATH = REPORT_DIR / "rag_retrieval_report.md"

INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(the\s+)?instructions", re.IGNORECASE),
    re.compile(r"override\s+(the\s+)?(policy|guardrails|instructions)", re.IGNORECASE),
)
APPLICANT_QUERY_PATTERN = re.compile(
    r"\b(applicant|customer|borrower)\b.{0,80}\b\d{6}\b|"
    r"\b\d{6}\b.{0,80}\b(risk|default|approve|decline|credit)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class KnowledgeChunk:
    """One source-grounded unit of retrievable documentation."""

    chunk_id: str
    source: str
    section: str
    text: str
    source_sha256: str


def load_rag_config(path: Path = RAG_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the version-controlled retrieval policy."""
    if not path.exists():
        raise FileNotFoundError(f"RAG configuration not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("trusted_sources"):
        raise ValueError("RAG configuration requires trusted_sources")
    return config


def sha256_text(text: str) -> str:
    """Return a deterministic UTF-8 content digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_prompt_injection(text: str) -> list[str]:
    """Return names of instruction-manipulation patterns found in text."""
    return [pattern.pattern for pattern in INJECTION_PATTERNS if pattern.search(text)]


def validate_knowledge_query(query: str) -> str:
    """Reject blank, injection-like, and applicant-specific retrieval queries."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Knowledge query must not be blank")
    if len(normalized) > 1000:
        raise ValueError("Knowledge query exceeds the maximum length")
    if detect_prompt_injection(normalized):
        raise ValueError("Knowledge query contains an instruction-manipulation pattern")
    if APPLICANT_QUERY_PATTERN.search(normalized):
        raise ValueError("Applicant-specific questions must use the governed inference workflow")
    return normalized


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into heading-aware sections."""
    sections: list[tuple[str, str]] = []
    heading = "Document"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if buffer and " ".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line.lstrip("#").strip() or "Untitled"
            buffer = []
        else:
            buffer.append(line)
    if buffer and " ".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _word_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Chunk text with deterministic word overlap."""
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Chunk size and overlap are invalid")
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def chunk_markdown(
    text: str,
    source: str,
    chunk_words: int = 180,
    overlap_words: int = 30,
) -> list[KnowledgeChunk]:
    """Create section-aware chunks with stable citations and source fingerprints."""
    source_hash = sha256_text(text)
    chunks = []
    sequence = 0
    for section, content in _markdown_sections(text):
        for part in _word_chunks(content, chunk_words, overlap_words):
            chunk_id = hashlib.sha256(f"{source}|{section}|{sequence}|{part}".encode()).hexdigest()[
                :16
            ]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source=source,
                    section=section,
                    text=part,
                    source_sha256=source_hash,
                )
            )
            sequence += 1
    return chunks


class LocalKnowledgeIndex:
    """Serializable local TF-IDF vector index with citation metadata."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix: Any, chunks: list[KnowledgeChunk]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def fit(cls, chunks: list[KnowledgeChunk]) -> LocalKnowledgeIndex:
        """Fit a deterministic unigram/bigram embedding index."""
        if not chunks:
            raise ValueError("At least one knowledge chunk is required")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        return cls(vectorizer, matrix, chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float = 0.02,
    ) -> list[dict[str, Any]]:
        """Retrieve ranked chunks with stable source citations."""
        query = validate_knowledge_query(query)
        if top_k <= 0 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        vector = self.vectorizer.transform([query])
        scores = linear_kernel(vector, self.matrix).reshape(-1)
        ranked = np.argsort(scores)[::-1]
        results = []
        for index in ranked:
            score = float(scores[index])
            if score < minimum_score:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "citation": f"[{chunk.source}#{chunk.section}]",
                    "score": round(score, 6),
                    "text": chunk.text,
                }
            )
            if len(results) == top_k:
                break
        return results


def build_knowledge_index(
    project_root: Path = PROJECT_ROOT,
    index_path: Path = RAG_INDEX_PATH,
    manifest_path: Path = RAG_MANIFEST_PATH,
    config_path: Path = RAG_CONFIG_PATH,
) -> dict[str, Any]:
    """Build an index from only explicit trusted, injection-scanned documents."""
    config = load_rag_config(config_path)
    index_config = config["index"]
    chunks: list[KnowledgeChunk] = []
    source_manifest = []
    for relative_source in config["trusted_sources"]:
        relative = Path(str(relative_source))
        source_path = (project_root / relative).resolve()
        if not source_path.is_relative_to(project_root.resolve()):
            raise ValueError(f"Trusted source escapes project root: {relative}")
        if not source_path.exists():
            raise FileNotFoundError(f"Trusted source is missing: {relative}")
        text = source_path.read_text(encoding="utf-8")
        findings = detect_prompt_injection(text)
        if findings and config["security"]["reject_prompt_injection_patterns"]:
            raise ValueError(f"Prompt-injection pattern detected in trusted source: {relative}")
        source_chunks = chunk_markdown(
            text,
            relative.as_posix(),
            chunk_words=int(index_config["chunk_words"]),
            overlap_words=int(index_config["chunk_overlap_words"]),
        )
        chunks.extend(source_chunks)
        source_manifest.append(
            {
                "source": relative.as_posix(),
                "sha256": sha256_text(text),
                "chunks": len(source_chunks),
            }
        )
    index = LocalKnowledgeIndex.fit(chunks)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, index_path)
    manifest = {
        "backend": index_config["backend"],
        "sources": source_manifest,
        "source_count": len(source_manifest),
        "chunk_count": len(chunks),
        "retrieved_text_is_untrusted_data": bool(
            config["security"]["retrieved_text_is_untrusted_data"]
        ),
        "applicant_queries_permitted": False,
        "limitations": config["limitations"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_knowledge_index(path: Path = RAG_INDEX_PATH) -> LocalKnowledgeIndex:
    """Load the local knowledge index after it has been built."""
    if not path.exists():
        raise FileNotFoundError("Run `risklens build-knowledge-index` first")
    index = joblib.load(path)
    if not isinstance(index, LocalKnowledgeIndex):
        raise TypeError("Knowledge index has an unexpected type")
    return index


def evaluate_knowledge_retrieval(
    index_path: Path = RAG_INDEX_PATH,
    evaluation_path: Path = RAG_EVALUATION_CONFIG_PATH,
    metrics_path: Path = RAG_EVALUATION_METRICS_PATH,
    report_path: Path = RAG_EVALUATION_REPORT_PATH,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate source hit rate and reciprocal rank on curated governance questions."""
    index = load_knowledge_index(index_path)
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    rows = []
    reciprocal_ranks = []
    for item in evaluation["queries"]:
        results = index.search(str(item["question"]), top_k=top_k, minimum_score=0.0)
        sources = [result["source"] for result in results]
        expected = str(item["expected_source"])
        rank = sources.index(expected) + 1 if expected in sources else None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        rows.append(
            {
                "question": item["question"],
                "expected_source": expected,
                "retrieved_sources": sources,
                "rank": rank,
                "hit_at_k": rank is not None,
            }
        )
    report = {
        "queries": len(rows),
        "top_k": top_k,
        "source_hit_rate_at_k": round(sum(row["hit_at_k"] for row in rows) / len(rows), 6),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks)), 6),
        "results": rows,
        "applicant_queries_evaluated": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# RiskLens AI RAG Retrieval Evaluation",
        "",
        f"- Queries: {report['queries']}",
        f"- Source hit rate@{top_k}: {report['source_hit_rate_at_k']:.2%}",
        f"- Mean reciprocal rank: {report['mean_reciprocal_rank']:.4f}",
        "- Applicant-specific questions: prohibited and not evaluated",
        "",
        "| Question | Expected source | Rank |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | `{row['expected_source']}` | "
            f"{row['rank'] if row['rank'] else 'miss'} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
