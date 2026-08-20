from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
MAX_TOP_K = 5
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CHINESE_STOP_BIGRAMS = {"什么", "怎么", "如何", "是否", "当前", "说明", "介绍"}


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    chunk_id: str
    title: str
    content: str

    def searchable_text(self) -> str:
        return f"{self.title}\n{self.content}"


def _tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for match in _TOKEN_PATTERN.findall(str(text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            if len(match) == 1:
                tokens.append(match)
                continue
            tokens.append(match)
            tokens.extend(
                match[index : index + 2]
                for index in range(len(match) - 1)
                if match[index : index + 2] not in _CHINESE_STOP_BIGRAMS
            )
        else:
            tokens.append(match)
    return tokens


def _markdown_chunks(path: Path) -> List[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    chunks: List[KnowledgeChunk] = []
    title = path.stem
    body: List[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if not content:
            return
        index = len(chunks) + 1
        chunks.append(
            KnowledgeChunk(
                source=path.name,
                chunk_id=f"{path.stem}#{index}",
                title=title,
                content=content,
            )
        )

    for line in text.splitlines():
        heading = _HEADING_PATTERN.match(line)
        if heading:
            flush()
            title = heading.group(2).strip()
            body = []
            continue
        body.append(line)
    flush()
    return chunks


class KnowledgeRetriever:
    """Dependency-free Markdown retriever with a replaceable search interface."""

    def __init__(self, knowledge_dir: Optional[Path] = None):
        self.knowledge_dir = Path(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)

    def load_chunks(self) -> List[KnowledgeChunk]:
        if not self.knowledge_dir.is_dir():
            return []
        chunks: List[KnowledgeChunk] = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            try:
                chunks.extend(_markdown_chunks(path))
            except (OSError, UnicodeError):
                continue
        return chunks

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        query_tokens = _tokens(str(query or "").strip())
        if not query_tokens:
            return []

        try:
            bounded_top_k = max(1, min(int(top_k), MAX_TOP_K))
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k 必须是整数。") from exc

        chunks = self.load_chunks()
        if not chunks:
            return []

        token_counts = [Counter(_tokens(chunk.searchable_text())) for chunk in chunks]
        document_frequency = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())

        average_length = sum(sum(counts.values()) for counts in token_counts) / len(token_counts)
        query_counts = Counter(query_tokens)
        scored = []
        for chunk, counts in zip(chunks, token_counts):
            length = max(1, sum(counts.values()))
            score = _bm25_score(
                query_counts,
                counts,
                document_frequency,
                document_count=len(chunks),
                document_length=length,
                average_length=max(1.0, average_length),
            )
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].source, item[1].chunk_id))
        return [
            {
                "source": chunk.source,
                "document": chunk.source,
                "chunk": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "score": round(score, 6),
            }
            for score, chunk in scored[:bounded_top_k]
        ]


def _bm25_score(
    query_counts: Counter,
    document_counts: Counter,
    document_frequency: Counter,
    *,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75
    for token, query_frequency in query_counts.items():
        term_frequency = document_counts.get(token, 0)
        if not term_frequency:
            continue
        frequency = document_frequency.get(token, 0)
        inverse_document_frequency = math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        denominator = term_frequency + k1 * (1 - b + b * document_length / average_length)
        score += inverse_document_frequency * term_frequency * (k1 + 1) / denominator * query_frequency
    return score


def retrieve_knowledge(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    return KnowledgeRetriever().search(query, top_k=top_k)
