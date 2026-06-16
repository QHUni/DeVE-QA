from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json

from .schema import Prediction, QAItem, TimeSpan, load_jsonl
from .text_utils import option_letter


@dataclass
class EvalResult:
    acc_qa: float
    miop: float
    iop_at_05: float
    miou: float
    iou_at_05: float
    acc_gqa: float
    count: int
    missing: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def pretty(self) -> str:
        return (f"N={self.count} missing={self.missing} | Acc@QA={self.acc_qa:.2f} | "
                f"mIoP={self.miop:.2f} IoP@0.5={self.iop_at_05:.2f} | "
                f"mIoU={self.miou:.2f} IoU@0.5={self.iou_at_05:.2f} | Acc@GQA={self.acc_gqa:.2f}")


def load_predictions(path: str | Path) -> Dict[str, Prediction]:
    rows = load_jsonl(path)
    preds = [Prediction.from_dict(row) for row in rows]
    return {p.qid: p for p in preds}


def evaluate(items: Iterable[QAItem], predictions: Mapping[str, Prediction], iop_threshold: float = 0.5) -> EvalResult:
    total = 0
    missing = 0
    qa_correct = 0
    gqa_correct = 0
    iops: List[float] = []
    ious: List[float] = []
    iop_hits = 0
    iou_hits = 0
    for item in items:
        total += 1
        pred = predictions.get(item.qid)
        if pred is None:
            missing += 1
            iops.append(0.0)
            ious.append(0.0)
            continue
        gold_answer = option_letter(item.answer or "", item.options) if item.answer else ""
        answer_ok = bool(gold_answer and pred.answer == gold_answer)
        qa_correct += int(answer_ok)
        if item.gt_span is not None:
            iop = pred.span.iop(item.gt_span)
            iou = pred.span.iou(item.gt_span)
        else:
            iop = 0.0
            iou = 0.0
        iops.append(iop)
        ious.append(iou)
        iop_hits += int(iop >= iop_threshold)
        iou_hits += int(iou >= iop_threshold)
        gqa_correct += int(answer_ok and iop >= iop_threshold)
    denom = max(1, total)
    return EvalResult(acc_qa=100.0 * qa_correct / denom,
                      miop=100.0 * sum(iops) / denom,
                      iop_at_05=100.0 * iop_hits / denom,
                      miou=100.0 * sum(ious) / denom,
                      iou_at_05=100.0 * iou_hits / denom,
                      acc_gqa=100.0 * gqa_correct / denom,
                      count=total, missing=missing)


def evaluate_by_group(items: Sequence[QAItem], predictions: Mapping[str, Prediction]) -> Dict[str, EvalResult]:
    groups: Dict[str, List[QAItem]] = {"short_video": [], "medium_video": [], "long_video": [], "single_event": [], "dense_event": []}
    for item in items:
        duration = item.duration or (item.gt_span.end if item.gt_span else 0.0)
        if duration < 60:
            groups["short_video"].append(item)
        elif duration < 180:
            groups["medium_video"].append(item)
        else:
            groups["long_video"].append(item)
        if len(item.captions) <= 2:
            groups["single_event"].append(item)
        else:
            groups["dense_event"].append(item)
    return {name: evaluate(rows, predictions) for name, rows in groups.items() if rows}


def confusion_matrix(items: Iterable[QAItem], predictions: Mapping[str, Prediction]) -> Dict[str, Dict[str, int]]:
    letters = list("ABCDE")
    matrix = {g: {p: 0 for p in letters} for g in letters}
    for item in items:
        if not item.answer:
            continue
        gold = option_letter(item.answer, item.options)
        pred = predictions.get(item.qid)
        if pred:
            matrix.setdefault(gold, {p: 0 for p in letters})[pred.answer] = matrix.setdefault(gold, {p: 0 for p in letters}).get(pred.answer, 0) + 1
    return matrix


def save_eval_report(path: str | Path, result: EvalResult, groups: Optional[Mapping[str, EvalResult]] = None) -> None:
    payload = {"overall": result.to_dict(), "groups": {k: v.to_dict() for k, v in (groups or {}).items()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_report(result: EvalResult, groups: Optional[Mapping[str, EvalResult]] = None) -> None:
    print(result.pretty())
    for name, group_result in (groups or {}).items():
        print(f"{name}: {group_result.pretty()}")


def predictions_from_list(preds: Sequence[Prediction]) -> Dict[str, Prediction]:
    return {p.qid: p for p in preds}


def error_cases(items: Iterable[QAItem], predictions: Mapping[str, Prediction], limit: int = 20) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in items:
        pred = predictions.get(item.qid)
        if pred is None:
            rows.append({"qid": item.qid, "error": "missing prediction"})
            continue
        gold = option_letter(item.answer or "", item.options) if item.answer else ""
        iop = pred.span.iop(item.gt_span) if item.gt_span else 0.0
        if pred.answer != gold or iop < 0.5:
            rows.append({"qid": item.qid, "question": item.question, "gold_answer": gold,
                         "pred_answer": pred.answer, "gold_span": item.gt_span.to_list() if item.gt_span else None,
                         "pred_span": pred.span.to_list(), "iop": iop, "rationale": pred.rationale})
        if len(rows) >= limit:
            break
    return rows
