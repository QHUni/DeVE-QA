from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import uuid


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("\r", " ").replace("\n", " ").strip()
    while "  " in value:
        value = value.replace("  ", " ")
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _option_dict(options: Any) -> Dict[str, str]:
    if isinstance(options, Mapping):
        return {str(k).strip().upper(): _clean_text(v) for k, v in options.items()}
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {alphabet[i]: _clean_text(v) for i, v in enumerate(options) if i < len(alphabet)}
    return {}


@dataclass
class TimeSpan:

    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        self.start = max(0.0, _float(self.start))
        self.end = max(self.start, _float(self.end, self.start))

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, other: "TimeSpan") -> float:
        left = max(self.start, other.start)
        right = min(self.end, other.end)
        return max(0.0, right - left)

    def iou(self, other: "TimeSpan") -> float:
        inter = self.overlap(other)
        union = self.duration + other.duration - inter
        return inter / union if union > 0 else 0.0

    def iop(self, other: "TimeSpan") -> float:
        inter = self.overlap(other)
        return inter / self.duration if self.duration > 0 else 0.0

    def clamp(self, duration: float) -> "TimeSpan":
        dur = max(0.0, _float(duration))
        return TimeSpan(min(self.start, dur), min(max(self.end, self.start), dur))

    def to_list(self) -> List[float]:
        return [round(self.start, 3), round(self.end, 3)]

    @classmethod
    def from_any(cls, value: Any) -> "TimeSpan":
        if isinstance(value, TimeSpan):
            return value
        if isinstance(value, Mapping):
            return cls(value.get("start", value.get("ts", 0.0)), value.get("end", value.get("te", 0.0)))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            return cls(value[0], value[1])
        return cls(0.0, 0.0)


@dataclass
class EventCaption:

    video_id: str
    text: str
    span: TimeSpan
    level: str = "short"
    source: str = "generated"
    score: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.video_id = _clean_text(self.video_id)
        self.text = _clean_text(self.text)
        self.span = TimeSpan.from_any(self.span)
        self.level = _clean_text(self.level) or "short"
        self.source = _clean_text(self.source) or "generated"
        self.score = _float(self.score, 1.0)

    def to_text_block(self) -> str:
        label = self.level[:1].upper()
        return f"{label}({self.span.start:.1f}-{self.span.end:.1f}s): {self.text}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["span"] = self.span.to_list()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], video_id: str = "") -> "EventCaption":
        return cls(
            video_id=_clean_text(data.get("video_id", video_id)),
            text=_clean_text(data.get("text", data.get("caption", data.get("sentence", "")))),
            span=TimeSpan.from_any(data.get("span", [data.get("start", 0), data.get("end", 0)])),
            level=_clean_text(data.get("level", "short")),
            source=_clean_text(data.get("source", "dataset")),
            score=_float(data.get("score", 1.0), 1.0),
            extra=dict(data.get("extra", {})) if isinstance(data.get("extra", {}), Mapping) else {},
        )


@dataclass
class QAItem:

    qid: str
    video_id: str
    question: str
    options: Dict[str, str]
    answer: Optional[str] = None
    gt_span: Optional[TimeSpan] = None
    video_path: Optional[str] = None
    duration: float = 0.0
    captions: List[EventCaption] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qid = _clean_text(self.qid) or uuid.uuid4().hex[:12]
        self.video_id = _clean_text(self.video_id) or self.qid
        self.question = _clean_text(self.question)
        self.options = _option_dict(self.options)
        self.answer = _clean_text(self.answer).upper() if self.answer is not None else None
        self.gt_span = TimeSpan.from_any(self.gt_span) if self.gt_span is not None else None
        self.video_path = _clean_text(self.video_path) if self.video_path else None
        self.duration = _float(self.duration, self.gt_span.end if self.gt_span else 0.0)
        self.captions = [c if isinstance(c, EventCaption) else EventCaption.from_dict(c, self.video_id) for c in self.captions]

    @property
    def answer_text(self) -> str:
        if self.answer and self.answer in self.options:
            return self.options[self.answer]
        return ""

    def option_block(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.options.items()))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["gt_span"] = self.gt_span.to_list() if self.gt_span else None
        data["captions"] = [c.to_dict() for c in self.captions]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QAItem":
        qid = data.get("qid", data.get("question_id", data.get("id", "")))
        video_id = data.get("video_id", data.get("video", data.get("video_name", "")))
        question = data.get("question", data.get("query", ""))
        options = data.get("options", data.get("candidates", data.get("choices", {})))
        answer = data.get("answer", data.get("label", data.get("correct", None)))
        span = data.get("gt_span", data.get("span", data.get("timestamps", None)))
        captions = data.get("captions", data.get("events", []))
        return cls(qid=qid, video_id=video_id, question=question, options=options, answer=answer,
                   gt_span=TimeSpan.from_any(span) if span is not None else None,
                   video_path=data.get("video_path", data.get("path", None)),
                   duration=data.get("duration", data.get("video_duration", 0.0)), captions=captions,
                   metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), Mapping) else {})


@dataclass
class Prediction:

    qid: str
    answer: str
    span: TimeSpan
    confidence: float = 0.0
    consistency: float = 0.0
    rationale: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qid = _clean_text(self.qid)
        self.answer = _clean_text(self.answer).upper()[:1]
        self.span = TimeSpan.from_any(self.span)
        self.confidence = _float(self.confidence)
        self.consistency = _float(self.consistency)
        self.rationale = _clean_text(self.rationale)

    def to_dict(self) -> Dict[str, Any]:
        return {"qid": self.qid, "answer": self.answer, "span": self.span.to_list(),
                "confidence": self.confidence, "consistency": self.consistency,
                "rationale": self.rationale, "raw": self.raw}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Prediction":
        return cls(data.get("qid", data.get("question_id", "")), data.get("answer", "A"),
                   TimeSpan.from_any(data.get("span", data.get("pred_span", [0, 0]))),
                   data.get("confidence", 0.0), data.get("consistency", 0.0),
                   data.get("rationale", ""), dict(data.get("raw", {})))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
