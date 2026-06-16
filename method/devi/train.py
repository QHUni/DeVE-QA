"""Training utilities for the runnable DeVi fallback ranker.

The SIGIR paper presents DeVi as a training-free MLLM composition.  This file
adds a complete supervised training path requested for code engineering: it
learns four scalar weights used by ``LexicalReasoningBackend`` to rank candidate
answers and event captions.  The objective is simple, deterministic, and fast;
it is useful for sanity checks, ablations, and local development before swapping
in heavy video-language models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random

from .dataset import DenseVideoQADataset, DatasetConfig
from .memory import TemporalEventMemory
from .schema import EventCaption, QAItem
from .text_utils import cosine, hashed_bow, jaccard, option_letter, softmax

FEATURES = ["q_cap", "a_cap", "qa_cap", "prior"]


@dataclass
class TrainConfig:
    qa_path: str
    captions_path: Optional[str] = None
    output_path: str = "outputs/ranker.json"
    epochs: int = 20
    lr: float = 0.3
    seed: int = 7
    max_items: int = 0
    l2: float = 0.001
    vector_dim: int = 512
    top_k_events: int = 20


class RankerTrainer:
    """Tiny pairwise logistic trainer for answer-event scoring."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.weights = {name: 1.0 / len(FEATURES) for name in FEATURES}
        self.memory = TemporalEventMemory()

    def _features(self, item: QAItem, answer_letter: str, caption: EventCaption) -> Dict[str, float]:
        ans = item.options.get(answer_letter, "")
        qv = hashed_bow(item.question, self.cfg.vector_dim)
        av = hashed_bow(ans, self.cfg.vector_dim)
        qav = hashed_bow(item.question + " " + ans, self.cfg.vector_dim)
        cv = hashed_bow(caption.text, self.cfg.vector_dim)
        return {"q_cap": cosine(qv, cv), "a_cap": cosine(av, cv),
                "qa_cap": cosine(qav, cv), "prior": jaccard(ans, caption.text)}

    def _score(self, feats: Mapping[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * feats.get(k, 0.0) for k in FEATURES)

    def _candidates(self, item: QAItem) -> List[Tuple[str, EventCaption, Dict[str, float]]]:
        selected, _ = self.memory.retrieve(item, self.cfg.top_k_events)
        rows: List[Tuple[str, EventCaption, Dict[str, float]]] = []
        for letter in item.options:
            for cap in selected:
                rows.append((letter, cap, self._features(item, letter, cap)))
        return rows

    def _best_gold(self, item: QAItem, rows: Sequence[Tuple[str, EventCaption, Dict[str, float]]]) -> Optional[Tuple[str, EventCaption, Dict[str, float]]]:
        if not item.answer:
            return None
        gold = option_letter(item.answer, item.options)
        gold_rows = [r for r in rows if r[0] == gold]
        if item.gt_span:
            gold_rows.sort(key=lambda r: (-r[1].span.iop(item.gt_span), -self._score(r[2])))
        else:
            gold_rows.sort(key=lambda r: -self._score(r[2]))
        return gold_rows[0] if gold_rows else None

    def _hard_negative(self, item: QAItem, rows: Sequence[Tuple[str, EventCaption, Dict[str, float]]]) -> Optional[Tuple[str, EventCaption, Dict[str, float]]]:
        gold = option_letter(item.answer or "", item.options) if item.answer else ""
        neg_rows = [r for r in rows if r[0] != gold]
        neg_rows.sort(key=lambda r: -self._score(r[2]))
        return neg_rows[0] if neg_rows else None

    def fit(self, dataset: DenseVideoQADataset) -> Dict[str, object]:
        for item in dataset:
            if item.video_id not in self.memory.entries:
                self.memory.build_for_item(item)
        history: List[Dict[str, float]] = []
        items = list(dataset.items)
        for epoch in range(1, self.cfg.epochs + 1):
            self.rng.shuffle(items)
            loss_sum = 0.0
            updates = 0
            for item in items:
                if not item.answer:
                    continue
                rows = self._candidates(item)
                pos = self._best_gold(item, rows)
                neg = self._hard_negative(item, rows)
                if pos is None or neg is None:
                    continue
                pos_score = self._score(pos[2])
                neg_score = self._score(neg[2])
                margin = pos_score - neg_score
                prob = 1.0 / (1.0 + math.exp(-max(-30, min(30, margin))))
                loss = -math.log(max(1e-8, prob))
                grad_coeff = (prob - 1.0)
                for name in FEATURES:
                    grad = grad_coeff * (pos[2].get(name, 0.0) - neg[2].get(name, 0.0)) + self.cfg.l2 * self.weights[name]
                    self.weights[name] -= self.cfg.lr * grad
                self._normalize_weights()
                loss_sum += loss
                updates += 1
            metrics = self.evaluate(dataset)
            metrics["epoch"] = epoch
            metrics["loss"] = loss_sum / max(1, updates)
            history.append(metrics)
        return {"weights": self.weights, "history": history, "config": asdict(self.cfg)}

    def _normalize_weights(self) -> None:
        for k in FEATURES:
            if not math.isfinite(self.weights[k]):
                self.weights[k] = 0.0
            self.weights[k] = max(-2.0, min(2.0, self.weights[k]))

    def predict_answer(self, item: QAItem) -> str:
        rows = self._candidates(item)
        if not rows:
            return next(iter(item.options.keys()), "A")
        best = max(rows, key=lambda r: self._score(r[2]))
        return best[0]

    def evaluate(self, dataset: DenseVideoQADataset) -> Dict[str, float]:
        total, correct = 0, 0
        for item in dataset:
            if not item.answer:
                continue
            total += 1
            correct += int(self.predict_answer(item) == option_letter(item.answer, item.options))
        return {"train_acc": correct / max(1, total)}

    def save(self, payload: Mapping[str, object], path: Optional[str] = None) -> None:
        out = Path(path or self.cfg.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def train_from_config(cfg: TrainConfig) -> Dict[str, object]:
    dataset = DenseVideoQADataset(DatasetConfig(qa_path=cfg.qa_path, captions_path=cfg.captions_path,
                                                max_items=cfg.max_items, shuffle=True, seed=cfg.seed,
                                                split="train"))
    trainer = RankerTrainer(cfg)
    payload = trainer.fit(dataset)
    trainer.save(payload)
    return payload


def load_ranker(path: str | Path) -> Dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    weights = data.get("weights", data)
    return {k: float(weights.get(k, 0.0)) for k in FEATURES}


def cli_main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train the lightweight DeVi answer-event ranker.")
    parser.add_argument("--qa", required=True, dest="qa_path")
    parser.add_argument("--captions", default=None, dest="captions_path")
    parser.add_argument("--output", default="outputs/ranker.json", dest="output_path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--max-items", type=int, default=0, dest="max_items")
    args = parser.parse_args()
    payload = train_from_config(TrainConfig(**vars(args)))
    print(json.dumps({"weights": payload["weights"], "last": payload["history"][-1]}, indent=2))


if __name__ == "__main__":
    cli_main()
