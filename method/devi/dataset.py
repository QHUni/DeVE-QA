from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import json
import random

from .schema import EventCaption, QAItem, TimeSpan, load_jsonl, save_jsonl
from .text_utils import normalize, option_letter


@dataclass
class DatasetConfig:
    qa_path: str
    captions_path: Optional[str] = None
    video_root: Optional[str] = None
    split: str = "test"
    max_items: int = 0
    seed: int = 7
    shuffle: bool = False


class DenseVideoQADataset:

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.qa_path = Path(cfg.qa_path)
        if not self.qa_path.exists():
            raise FileNotFoundError(f"QA file not found: {self.qa_path}")
        self.caption_bank = self._load_caption_bank(cfg.captions_path)
        self.items = self._load_items()
        if cfg.shuffle:
            rng = random.Random(cfg.seed)
            rng.shuffle(self.items)
        if cfg.max_items and cfg.max_items > 0:
            self.items = self.items[: cfg.max_items]

    def _load_caption_bank(self, captions_path: Optional[str]) -> Dict[str, List[EventCaption]]:
        if not captions_path:
            return {}
        path = Path(captions_path)
        if not path.exists():
            raise FileNotFoundError(f"Captions file not found: {path}")
        rows = load_jsonl(path) if path.suffix.lower() == ".jsonl" else json.loads(path.read_text())
        bank: Dict[str, List[EventCaption]] = {}
        if isinstance(rows, Mapping):
            iterator = rows.items()
            for vid, caps in iterator:
                bank[str(vid)] = [EventCaption.from_dict(c, str(vid)) for c in caps]
        else:
            for row in rows:
                vid = str(row.get("video_id", row.get("video", "")))
                if not vid:
                    continue
                if "captions" in row and isinstance(row["captions"], list):
                    caps = [EventCaption.from_dict(c, vid) for c in row["captions"]]
                else:
                    caps = [EventCaption.from_dict(row, vid)]
                bank.setdefault(vid, []).extend(caps)
        return bank

    def _load_items(self) -> List[QAItem]:
        rows = load_jsonl(self.qa_path) if self.qa_path.suffix.lower() == ".jsonl" else json.loads(self.qa_path.read_text())
        if isinstance(rows, Mapping):
            rows = rows.get("data", rows.get("items", []))
        items: List[QAItem] = []
        for idx, row in enumerate(rows):
            item = self._parse_row(row, idx)
            if item.video_id in self.caption_bank:
                item.captions.extend(self.caption_bank[item.video_id])
            if self.cfg.video_root and item.video_path:
                p = Path(item.video_path)
                if not p.is_absolute():
                    item.video_path = str(Path(self.cfg.video_root) / p)
            items.append(item)
        return items

    def _parse_row(self, row: Mapping[str, object], idx: int) -> QAItem:
        if "question" in row or "query" in row:
            item = QAItem.from_dict(row)
        else:
            item = self._parse_legacy_row(row, idx)
        if not item.qid:
            item.qid = f"{self.cfg.split}_{idx:07d}"
        if item.answer:
            item.answer = option_letter(item.answer, item.options)
        return item

    def _parse_legacy_row(self, row: Mapping[str, object], idx: int) -> QAItem:
        q = row.get("q", row.get("Q", ""))
        opts = row.get("a", row.get("answers", row.get("options", {})))
        answer = row.get("correct_idx", row.get("answer_idx", row.get("answer", None)))
        if isinstance(answer, int):
            answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer]
        return QAItem(qid=str(row.get("id", idx)), video_id=str(row.get("vid", row.get("video_id", idx))),
                      question=str(q), options=opts, answer=answer,
                      gt_span=TimeSpan.from_any(row.get("time", row.get("span", [0, 0]))),
                      video_path=row.get("video_path", None), duration=float(row.get("duration", 0) or 0),
                      captions=row.get("events", row.get("captions", [])),
                      metadata={"legacy": True})

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[QAItem]:
        return iter(self.items)

    def __getitem__(self, index: int) -> QAItem:
        return self.items[index]

    def by_video(self) -> Dict[str, List[QAItem]]:
        groups: Dict[str, List[QAItem]] = {}
        for item in self.items:
            groups.setdefault(item.video_id, []).append(item)
        return groups

    def gold_rows(self) -> List[Dict[str, object]]:
        return [item.to_dict() for item in self.items]

    def save_normalized(self, path: str | Path) -> None:
        save_jsonl(path, self.gold_rows())


def make_sample_dataset(out_dir: str | Path) -> Tuple[Path, Path]:
    """Create a tiny deterministic dataset for smoke tests and examples."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qa_rows = [
        {
            "qid": "sample_001", "video_id": "video_cat", "question": "What is the man doing to the black cat?",
            "options": {"A": "Puts the cat on the floor", "B": "Sits next to the cat", "C": "Reaches for scissors", "D": "Cutting its claws", "E": "Puts it in the kitchen"},
            "answer": "D", "gt_span": [30, 55], "duration": 70, "video_path": "video_cat.mp4",
        },
        {
            "qid": "sample_002", "video_id": "video_ski", "question": "Why do they need the intro before the ski tricks?",
            "options": {"A": "Present several landscapes and people skiing", "B": "Check camera settings", "C": "Clean the road", "D": "Show cooking steps", "E": "Repair a bicycle"},
            "answer": "A", "gt_span": [0, 39.3], "duration": 153.5, "video_path": "video_ski.mp4",
        },
        {
            "qid": "sample_003", "video_id": "video_bike", "question": "Why is the crowd shouting when the biker is on top of the dirt dome?",
            "options": {"A": "The biker finished perfectly", "B": "Worrying that the biker will fall", "C": "Arguing with people", "D": "Complaining about noise", "E": "They are disappointed"},
            "answer": "B", "gt_span": [88.1, 115.3], "duration": 130.6, "video_path": "video_bike.mp4",
        },
    ]
    cap_rows = [
        {"video_id": "video_cat", "captions": [
            {"text": "A man stands next to a black cat and fetches something while the cat moves around.", "span": [0, 30], "level": "short"},
            {"text": "An old man holds scissors and carefully cuts the black cat's claws.", "span": [30, 55], "level": "short"},
            {"text": "The cat is put down after the claw cutting is finished.", "span": [55, 70], "level": "medium"},
        ]},
        {"video_id": "video_ski", "captions": [
            {"text": "A group presents landscapes and skiing skills in an introductory title sequence.", "span": [0, 39.3], "level": "long"},
            {"text": "A man talks to the camera about ski tricks.", "span": [39.3, 66.8], "level": "medium"},
            {"text": "Young boys perform ski tricks and speak to the camera in snowy landscapes.", "span": [66.8, 153.5], "level": "long"},
        ]},
        {"video_id": "video_bike", "captions": [
            {"text": "A biker performs an obstacle course in a stadium with a watching crowd.", "span": [0, 88.1], "level": "long"},
            {"text": "The biker rides on top of a giant dome of dirt and the crowd shouts with concern.", "span": [88.1, 115.3], "level": "short"},
            {"text": "After finishing the race, the biker appears very excited.", "span": [115.3, 130.6], "level": "short"},
        ]},
    ]
    qa_path = out / "qa.jsonl"
    cap_path = out / "captions.jsonl"
    save_jsonl(qa_path, qa_rows)
    save_jsonl(cap_path, cap_rows)
    return qa_path, cap_path


def load_dataset_from_args(args: object) -> DenseVideoQADataset:
    cfg = DatasetConfig(qa_path=getattr(args, "qa", ""), captions_path=getattr(args, "captions", None),
                        video_root=getattr(args, "video_root", None), split=getattr(args, "split", "test"),
                        max_items=int(getattr(args, "max_items", 0) or 0), seed=int(getattr(args, "seed", 7) or 7),
                        shuffle=bool(getattr(args, "shuffle", False)))
    return DenseVideoQADataset(cfg)


def describe_dataset(dataset: DenseVideoQADataset) -> Dict[str, object]:
    videos = dataset.by_video()
    answers = {k: 0 for k in "ABCDE"}
    cap_count = 0
    durations: List[float] = []
    for item in dataset:
        if item.answer in answers:
            answers[item.answer] += 1
        cap_count += len(item.captions)
        if item.duration:
            durations.append(item.duration)
    return {"num_questions": len(dataset), "num_videos": len(videos), "answer_hist": answers,
            "avg_captions_per_question": cap_count / max(1, len(dataset)),
            "avg_duration": sum(durations) / max(1, len(durations))}
