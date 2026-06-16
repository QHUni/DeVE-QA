from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import argparse
import json
import sys
import time

from .dataset import DenseVideoQADataset, DatasetConfig, describe_dataset
from .eval import evaluate, predictions_from_list, print_report, save_eval_report
from .model import DeViConfig, DeViModel, MockCaptionBackend
from .openai_backend import make_openai_backends
from .schema import Prediction, QAItem


def build_model_from_args(args: argparse.Namespace) -> DeViModel:
    cfg = DeViConfig(short_len=args.short_len, medium_len=args.medium_len, long_len=args.long_len,
                    top_k_events=args.top_k_events, consistency_threshold=args.consistency_threshold,
                    max_verify_iters=args.max_verify_iters, use_dataset_captions=not args.ignore_dataset_captions,
                    ranker_path=args.ranker, seed=args.seed)
    caption_backend = None
    reasoning_backend = None
    if args.llm_backend == "openai" or args.caption_backend == "openai":
        caption_backend, reasoning_backend = make_openai_backends(
            model=args.openai_model,
            use_openai_captioner=(args.caption_backend == "openai"),
            api_key_env=args.openai_api_key_env,
            base_url=args.openai_base_url or None,
            max_output_tokens=args.openai_max_output_tokens,
            caption_frame_count=args.openai_caption_frames,
        )
    if args.caption_backend == "mock":
        caption_backend = MockCaptionBackend()
    return DeViModel(cfg, caption_backend=caption_backend, reasoning_backend=reasoning_backend)


def run_inference(args: argparse.Namespace) -> List[Prediction]:
    dataset = DenseVideoQADataset(DatasetConfig(qa_path=args.qa, captions_path=args.captions,
                                                video_root=args.video_root, split=args.split,
                                                max_items=args.max_items, seed=args.seed,
                                                shuffle=False))
    model = build_model_from_args(args)
    start = time.time()
    preds: List[Prediction] = []
    for idx, item in enumerate(dataset, 1):
        pred = model.predict_one(item)
        preds.append(pred)
        if args.verbose:
            print(f"[{idx}/{len(dataset)}] {item.qid} -> {pred.answer} {pred.span.to_list()} consistency={pred.consistency:.3f}")
    elapsed = time.time() - start
    model.save_predictions(args.output, preds)
    if args.eval:
        result = evaluate(dataset, predictions_from_list(preds))
        print_report(result)
        save_eval_report(Path(args.output).with_suffix(".metrics.json"), result)
    print(json.dumps({"output": args.output, "num_predictions": len(preds), "seconds": elapsed,
                      "dataset": describe_dataset(dataset)}, ensure_ascii=False, indent=2))
    return preds


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeVi dense-event VideoQA inference.")
    parser.add_argument("--qa", required=True, help="Path to QA JSONL/JSON.")
    parser.add_argument("--captions", default=None, help="Optional captions JSONL/JSON.")
    parser.add_argument("--video-root", default=None, help="Root path for relative videos.")
    parser.add_argument("--output", default="outputs/predictions.jsonl", help="Prediction JSONL path.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short-len", type=float, default=15.0)
    parser.add_argument("--medium-len", type=float, default=35.0)
    parser.add_argument("--long-len", type=float, default=65.0)
    parser.add_argument("--top-k-events", type=int, default=10)
    parser.add_argument("--consistency-threshold", type=float, default=0.40)
    parser.add_argument("--max-verify-iters", type=int, default=2)
    parser.add_argument("--ranker", default=None, help="Optional trained ranker JSON.")
    parser.add_argument("--llm-backend", choices=["lexical", "openai"], default="lexical",
                        help="Use lexical local reasoner or GPT-4o/OpenAI API reasoner.")
    parser.add_argument("--caption-backend", choices=["mock", "openai"], default="mock",
                        help="Caption clips with local mock text or GPT-4o vision. Dataset captions are still used unless --ignore-dataset-captions is set.")
    parser.add_argument("--openai-model", default="gpt-4o", help="OpenAI model name for zero-shot DeVi backbone.")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the OpenAI API key.")
    parser.add_argument("--openai-base-url", default="", help="Optional custom OpenAI-compatible base URL.")
    parser.add_argument("--openai-max-output-tokens", type=int, default=900, help="Max output tokens for GPT-4o JSON answers/captions.")
    parser.add_argument("--openai-caption-frames", type=int, default=8, help="Number of frames sampled per clip for GPT-4o captioning.")
    parser.add_argument("--ignore-dataset-captions", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Evaluate if gold labels and spans exist.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def cli_main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    cli_main()

INFERENCE_NOTES = """
Inference mirrors the DeVi paper pipeline.  First, each item is converted into
hierarchical event captions.  In the included local mode, existing dataset
captions are reused; otherwise, MockCaptionBackend creates deterministic segment
captions so the code path remains executable.  Second, TemporalEventMemory
contextualizes captions and retrieves the top-k relevant events for the current
question.  Third, the reasoning backend predicts an answer option and a temporal
span.  Fourth, self-consistency checking compares the answer text with captions
inside the predicted span and can trigger verification iterations.

For real experiments, keep this command-line file unchanged and replace backend
classes in model.py.  The recommended workflow is to precompute captions per
video, save them as JSONL, and pass --captions during inference.  This avoids
re-captioning videos for every question, which is important for DeVE-QA because
multiple questions share the same dense-event video.
"""


def inference_notes() -> str:
    return INFERENCE_NOTES


def default_command_example() -> str:
    return (
        "python -m devi.infer --qa data/sample/qa.jsonl "
        "--captions data/sample/captions.jsonl --output outputs/predictions.jsonl --eval"
    )


