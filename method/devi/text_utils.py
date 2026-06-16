from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import hashlib
import json
import math
import re

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "being", "been", "what", "why", "how",
    "when", "where", "who", "which", "does", "do", "did", "this", "that", "there",
    "from", "by", "as", "after", "before", "during", "video", "scene", "person",
}


def normalize(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def char_ngrams(text: str, n: int = 3) -> List[str]:
    clean = re.sub(r"\s+", " ", normalize(text))
    if len(clean) <= n:
        return [clean] if clean else []
    return [clean[i:i + n] for i in range(len(clean) - n + 1)]


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def hashed_bow(text: str, dim: int = 512, use_char: bool = True) -> List[float]:
    vec = [0.0] * dim
    terms = tokenize(text)
    for tok in terms:
        vec[stable_hash("w:" + tok, dim)] += 1.0
    if use_char:
        for gram in char_ngrams(text):
            vec[stable_hash("c:" + gram, dim)] += 0.2
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(x) * float(x) for x in a[:n]))
    nb = math.sqrt(sum(float(x) * float(x) for x in b[:n]))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def option_letter(value: str, options: Mapping[str, str]) -> str:
    value = (value or "").strip().upper()
    if value in options:
        return value
    for key, text in options.items():
        if normalize(value) == normalize(text):
            return key
    if value and value[0] in options:
        return value[0]
    return next(iter(options.keys()), "A")


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\s*;\s*", text or "")
    return [p.strip() for p in parts if p.strip()]


def summarize_texts(texts: Iterable[str], max_sentences: int = 6) -> str:
    sentences: List[str] = []
    seen = set()
    for text in texts:
        for sent in split_sentences(text):
            key = normalize(sent)
            if key and key not in seen:
                seen.add(key)
                sentences.append(sent)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences)


@dataclass
class TinyTfidfVectorizer:

    min_df: int = 1
    max_features: int = 5000
    vocab: Dict[str, int] = field(default_factory=dict)
    idf: List[float] = field(default_factory=list)

    def fit(self, documents: Iterable[str]) -> "TinyTfidfVectorizer":
        docs = [tokenize(doc) for doc in documents]
        df: MutableMapping[str, int] = defaultdict(int)
        for tokens in docs:
            for tok in set(tokens):
                df[tok] += 1
        terms = [(tok, c) for tok, c in df.items() if c >= self.min_df]
        terms.sort(key=lambda x: (-x[1], x[0]))
        terms = terms[: self.max_features]
        self.vocab = {tok: i for i, (tok, _) in enumerate(terms)}
        n_docs = max(1, len(docs))
        self.idf = [0.0] * len(self.vocab)
        for tok, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
        return self

    def transform_one(self, document: str) -> List[float]:
        vec = [0.0] * len(self.vocab)
        counts = Counter(tokenize(document))
        total = sum(counts.values()) or 1
        for tok, count in counts.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] = (count / total) * self.idf[idx]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    def transform(self, documents: Iterable[str]) -> List[List[float]]:
        return [self.transform_one(doc) for doc in documents]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump({"min_df": self.min_df, "max_features": self.max_features,
                       "vocab": self.vocab, "idf": self.idf}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TinyTfidfVectorizer":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data.get("min_df", 1), data.get("max_features", 5000),
                   {str(k): int(v) for k, v in data.get("vocab", {}).items()},
                   [float(x) for x in data.get("idf", [])])


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    temp = max(1e-6, temperature)
    shifted = [s / temp for s in scores]
    m = max(shifted)
    exps = [math.exp(s - m) for s in shifted]
    total = sum(exps) or 1.0
    return [e / total for e in exps]
