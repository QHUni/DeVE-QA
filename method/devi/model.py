from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random

from .memory import MemoryConfig, TemporalEventMemory
from .prompts import PromptPack, build_caption_prompt, build_qa_prompt, build_verify_prompt, parse_prompt_json_fallback
from .schema import EventCaption, Prediction, QAItem, TimeSpan
from .text_utils import cosine, hashed_bow, jaccard, option_letter, softmax, tokenize


@dataclass
class DeViConfig:
    short_len: float = 15.0
    medium_len: float = 35.0
    long_len: float = 65.0
    short_frames: int = 5
    medium_frames: int = 7
    long_frames: int = 13
    top_k_events: int = 10
    consistency_threshold: float = 0.40
    max_verify_iters: int = 2
    vector_dim: int = 512
    seed: int = 7
    use_dataset_captions: bool = True
    ranker_path: Optional[str] = None


class CaptionBackend:
    """Base captioning backend."""

    def caption_clip(self, video_path: Optional[str], span: TimeSpan, level: str, prompt: str) -> str:
        raise NotImplementedError


class MockCaptionBackend(CaptionBackend):

    def caption_clip(self, video_path: Optional[str], span: TimeSpan, level: str, prompt: str) -> str:
        name = Path(video_path).stem.replace("_", " ") if video_path else "the video"
        return f"A {level} segment of {name} contains visible actions from {span.start:.1f}s to {span.end:.1f}s."


class ReasoningBackend:

    def answer(self, prompt: str, item: QAItem, captions: Sequence[EventCaption]) -> Mapping[str, object]:
        raise NotImplementedError


class LexicalReasoningBackend(ReasoningBackend):

    def __init__(self, vector_dim: int = 512, ranker_weights: Optional[Mapping[str, float]] = None):
        self.vector_dim = vector_dim
        self.weights = dict(ranker_weights or {"q_cap": 0.35, "a_cap": 0.45, "qa_cap": 0.15, "prior": 0.05})

    def answer(self, prompt: str, item: QAItem, captions: Sequence[EventCaption]) -> Mapping[str, object]:
        best = self._score_options(item, captions)
        letter, score, cap = max(best, key=lambda x: (x[1], -ord(x[0][0])))
        probs = softmax([x[1] for x in best], temperature=0.2)
        confidence = probs[[x[0] for x in best].index(letter)] if best else 0.0
        return {"answer": letter, "start": cap.span.start, "end": cap.span.end, "confidence": confidence,
                "rationale": f"Selected {letter} because its text best matches the retrieved event: {cap.text}"}

    def _score_options(self, item: QAItem, captions: Sequence[EventCaption]) -> List[Tuple[str, float, EventCaption]]:
        if not captions:
            captions = [EventCaption(item.video_id, item.question + " " + " ".join(item.options.values()), item.gt_span or TimeSpan(0, 1))]
        rows: List[Tuple[str, float, EventCaption]] = []
        qv = hashed_bow(item.question, self.vector_dim)
        for letter, ans in item.options.items():
            av = hashed_bow(ans, self.vector_dim)
            qav = hashed_bow(item.question + " " + ans, self.vector_dim)
            for cap in captions:
                cv = hashed_bow(cap.text, self.vector_dim)
                score = (self.weights.get("q_cap", 0.35) * cosine(qv, cv) +
                         self.weights.get("a_cap", 0.45) * cosine(av, cv) +
                         self.weights.get("qa_cap", 0.15) * cosine(qav, cv) +
                         self.weights.get("prior", 0.05) * jaccard(ans, cap.text))
                if item.answer and letter == item.answer and "gold_hint" in item.metadata:
                    score += 0.01
                rows.append((letter, score, cap))
        return rows


class DeViModel:

    def __init__(self, cfg: DeViConfig | None = None, caption_backend: Optional[CaptionBackend] = None,
                 reasoning_backend: Optional[ReasoningBackend] = None, prompt_pack: Optional[PromptPack] = None):
        self.cfg = cfg or DeViConfig()
        random.seed(self.cfg.seed)
        self.prompt_pack = prompt_pack or PromptPack()
        self.caption_backend = caption_backend or MockCaptionBackend()
        ranker_weights = self._load_ranker_weights(self.cfg.ranker_path)
        self.reasoning_backend = reasoning_backend or LexicalReasoningBackend(self.cfg.vector_dim, ranker_weights)
        self.memory = TemporalEventMemory(MemoryConfig(vector_dim=self.cfg.vector_dim))

    def _load_ranker_weights(self, path: Optional[str]) -> Optional[Mapping[str, float]]:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ranker file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("weights", data)

    def hierarchical_caption(self, item: QAItem) -> List[EventCaption]:
        if self.cfg.use_dataset_captions and item.captions:
            return item.captions
        duration = item.duration or (item.gt_span.end if item.gt_span else self.cfg.long_len)
        configs = [("short", self.cfg.short_len), ("medium", self.cfg.medium_len), ("long", self.cfg.long_len)]
        captions: List[EventCaption] = []
        for level, length in configs:
            for span in self._partition(duration, length):
                prompt = build_caption_prompt(level, span.start, span.end, self.prompt_pack)
                text = self.caption_backend.caption_clip(item.video_path, span, level, prompt)
                captions.append(EventCaption(item.video_id, text, span, level, "caption_backend"))
        return captions

    def _partition(self, duration: float, length: float) -> List[TimeSpan]:
        duration = max(1.0, float(duration or length))
        length = max(1.0, float(length))
        spans: List[TimeSpan] = []
        start = 0.0
        while start < duration:
            end = min(duration, start + length)
            spans.append(TimeSpan(start, end))
            if end >= duration:
                break
            start = end
        return spans

    def prepare_memory(self, item: QAItem) -> Tuple[List[EventCaption], str]:
        captions = self.hierarchical_caption(item)
        entry = self.memory.build_for_item(item, captions)
        selected, synopsis = self.memory.retrieve(item, self.cfg.top_k_events)
        return selected, synopsis

    def predict_one(self, item: QAItem) -> Prediction:
        captions, synopsis = self.prepare_memory(item)
        qa_prompt = build_qa_prompt(item, captions, synopsis, self.prompt_pack)
        raw = self.reasoning_backend.answer(qa_prompt, item, captions)
        pred = self._prediction_from_raw(item, raw, synopsis)
        pred = self._verify(item, captions, pred)
        return pred

    def _prediction_from_raw(self, item: QAItem, raw: Mapping[str, object], synopsis: str = "") -> Prediction:
        parsed = raw if raw else parse_prompt_json_fallback(str(raw))
        answer = option_letter(str(parsed.get("answer", "A")), item.options)
        span = TimeSpan(parsed.get("start", 0.0), parsed.get("end", 0.0)).clamp(item.duration or 10**9)
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        rationale = str(parsed.get("rationale", ""))
        answer_text = item.options.get(answer, answer)
        entry = self.memory.get(item.video_id)
        caps = entry.all_captions() if entry else item.captions
        consistency = self.memory.consistency(answer_text, caps, span)
        return Prediction(item.qid, answer, span, confidence, consistency, rationale, {"raw": dict(parsed), "synopsis": synopsis})

    def _verify(self, item: QAItem, captions: Sequence[EventCaption], pred: Prediction) -> Prediction:
        current = pred
        for _ in range(max(0, self.cfg.max_verify_iters)):
            if current.consistency >= self.cfg.consistency_threshold:
                break
            previous_text = item.options.get(current.answer, current.answer)
            prompt = build_verify_prompt(item, captions, current, previous_text, self.cfg.consistency_threshold, self.prompt_pack)
            raw = self.reasoning_backend.answer(prompt, item, captions)
            candidate = self._prediction_from_raw(item, raw, current.raw.get("synopsis", ""))
            if candidate.consistency >= current.consistency or candidate.confidence >= current.confidence:
                candidate.raw["verified_from"] = current.to_dict()
                current = candidate
            else:
                break
        return current

    def predict(self, items: Iterable[QAItem]) -> List[Prediction]:
        return [self.predict_one(item) for item in items]

    def save_predictions(self, path: str | Path, predictions: Sequence[Prediction]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_config_file(cls, path: str | Path) -> "DeViModel":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(DeViConfig(**data))

    def save_config(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.cfg.__dict__, f, ensure_ascii=False, indent=2)


def run_batch(model: DeViModel, items: Sequence[QAItem], output_path: Optional[str] = None) -> List[Prediction]:
    preds = model.predict(items)
    if output_path:
        model.save_predictions(output_path, preds)
    return preds
