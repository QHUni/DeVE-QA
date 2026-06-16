from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math

from .schema import EventCaption, QAItem, TimeSpan
from .text_utils import TinyTfidfVectorizer, cosine, hashed_bow, jaccard, summarize_texts, tokenize


@dataclass
class MemoryConfig:
    vector_dim: int = 512
    max_captions: int = 200
    contextualize: bool = True
    synopsis_sentences: int = 8
    level_weights: Dict[str, float] = field(default_factory=lambda: {"short": 1.05, "medium": 1.0, "long": 0.95, "synopsis": 0.8})


@dataclass
class MemoryEntry:
    video_id: str
    captions: List[EventCaption] = field(default_factory=list)
    contextualized: List[EventCaption] = field(default_factory=list)
    synopsis: str = ""
    features: List[List[float]] = field(default_factory=list)

    def all_captions(self) -> List[EventCaption]:
        return self.contextualized if self.contextualized else self.captions

    def to_dict(self) -> Dict[str, object]:
        return {"video_id": self.video_id, "captions": [c.to_dict() for c in self.captions],
                "contextualized": [c.to_dict() for c in self.contextualized], "synopsis": self.synopsis,
                "features": self.features}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MemoryEntry":
        vid = str(data.get("video_id", ""))
        return cls(video_id=vid,
                   captions=[EventCaption.from_dict(c, vid) for c in data.get("captions", [])],
                   contextualized=[EventCaption.from_dict(c, vid) for c in data.get("contextualized", [])],
                   synopsis=str(data.get("synopsis", "")),
                   features=[[float(x) for x in row] for row in data.get("features", [])])


class TemporalEventMemory:
    def __init__(self, cfg: MemoryConfig | None = None):
        self.cfg = cfg or MemoryConfig()
        self.entries: Dict[str, MemoryEntry] = {}

    def build_for_item(self, item: QAItem, captions: Optional[Sequence[EventCaption]] = None) -> MemoryEntry:
        caps = list(captions if captions is not None else item.captions)
        if not caps:
            caps = self._fallback_captions(item)
        caps = sorted(caps, key=lambda c: (c.span.start, c.span.end, c.level))[: self.cfg.max_captions]
        entry = MemoryEntry(video_id=item.video_id, captions=caps)
        entry.contextualized = self.contextualize(caps, item.question) if self.cfg.contextualize else caps
        entry.synopsis = self.make_synopsis(entry.contextualized)
        entry.features = [hashed_bow(c.text, self.cfg.vector_dim) for c in entry.contextualized]
        self.entries[item.video_id] = entry
        return entry

    def _fallback_captions(self, item: QAItem) -> List[EventCaption]:
        span = item.gt_span or TimeSpan(0, item.duration or 1.0)
        guess = " ".join([item.question] + list(item.options.values()))
        return [EventCaption(item.video_id, f"Uncaptioned video segment related to: {guess}", span, "long", "fallback")]

    def contextualize(self, captions: Sequence[EventCaption], question: str = "") -> List[EventCaption]:
        all_text = " ".join(c.text for c in captions)
        q_terms = set(tokenize(question))
        contextualized: List[EventCaption] = []
        for cap in captions:
            related_terms = [t for t in tokenize(all_text) if t in q_terms]
            prefix = ""
            if related_terms:
                prefix = "In context of " + ", ".join(sorted(set(related_terms))[:5]) + ", "
            text = cap.text
            if prefix and not text.lower().startswith("in context"):
                text = prefix + text[0].lower() + text[1:]
            contextualized.append(EventCaption(cap.video_id, text, cap.span, cap.level, "contextualized", cap.score, cap.extra))
        return contextualized

    def make_synopsis(self, captions: Sequence[EventCaption]) -> str:
        return summarize_texts((c.text for c in captions), self.cfg.synopsis_sentences)

    def get(self, video_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(video_id)

    def retrieve(self, item: QAItem, top_k: int = 8) -> Tuple[List[EventCaption], str]:
        entry = self.entries.get(item.video_id) or self.build_for_item(item)
        query = item.question + " " + " ".join(item.options.values())
        qv = hashed_bow(query, self.cfg.vector_dim)
        scored: List[Tuple[float, EventCaption]] = []
        for idx, cap in enumerate(entry.all_captions()):
            cv = entry.features[idx] if idx < len(entry.features) else hashed_bow(cap.text, self.cfg.vector_dim)
            level_weight = self.cfg.level_weights.get(cap.level, 1.0)
            score = (0.75 * cosine(qv, cv) + 0.25 * jaccard(query, cap.text)) * level_weight
            scored.append((score, cap))
        scored.sort(key=lambda x: (-x[0], x[1].span.start))
        selected = [cap for _, cap in scored[: max(1, top_k)]]
        selected.sort(key=lambda c: c.span.start)
        return selected, entry.synopsis

    def consistency(self, answer_text: str, captions: Sequence[EventCaption], span: TimeSpan) -> float:
        if not captions:
            return 0.0
        answer_vec = hashed_bow(answer_text, self.cfg.vector_dim)
        scores: List[float] = []
        for cap in captions:
            overlap = span.iou(cap.span)
            if overlap <= 0 and span.duration > 0:
                center = (span.start + span.end) / 2
                if cap.span.start <= center <= cap.span.end:
                    overlap = 0.2
            text_sim = cosine(answer_vec, hashed_bow(cap.text, self.cfg.vector_dim))
            scores.append((0.5 + overlap) * text_sim)
        return max(scores) if scores else 0.0

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump({"cfg": self.cfg.__dict__, "entries": {k: v.to_dict() for k, v in self.entries.items()}},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TemporalEventMemory":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg_data = dict(data.get("cfg", {}))
        cfg_data["level_weights"] = dict(cfg_data.get("level_weights", {"short": 1.05, "medium": 1.0, "long": 0.95, "synopsis": 0.8}))
        mem = cls(MemoryConfig(**cfg_data))
        mem.entries = {k: MemoryEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
        return mem

    def merge(self, other: "TemporalEventMemory") -> None:
        for vid, entry in other.entries.items():
            if vid not in self.entries:
                self.entries[vid] = entry
            else:
                current = self.entries[vid]
                current.captions.extend(entry.captions)
                current.contextualized.extend(entry.contextualized)
                current.synopsis = self.make_synopsis(current.contextualized or current.captions)
                current.features = [hashed_bow(c.text, self.cfg.vector_dim) for c in current.all_captions()]

    def stats(self) -> Dict[str, float]:
        n_videos = len(self.entries)
        n_caps = sum(len(e.all_captions()) for e in self.entries.values())
        avg_caps = n_caps / max(1, n_videos)
        return {"videos": float(n_videos), "captions": float(n_caps), "avg_captions": avg_caps}


def build_memory_for_dataset(items: Iterable[QAItem], cfg: MemoryConfig | None = None) -> TemporalEventMemory:
    memory = TemporalEventMemory(cfg)
    seen = set()
    for item in items:
        if item.video_id not in seen:
            memory.build_for_item(item)
            seen.add(item.video_id)
    return memory
