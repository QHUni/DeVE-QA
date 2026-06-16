from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from devi.dataset import DatasetConfig, DenseVideoQADataset, describe_dataset, make_sample_dataset
from devi.eval import evaluate, evaluate_by_group, load_predictions, print_report, save_eval_report
from devi.infer import make_parser as make_infer_parser, run_inference
from devi.train import TrainConfig, train_from_config


def cmd_make_sample(args: argparse.Namespace) -> None:
    qa, caps = make_sample_dataset(args.output_dir)
    print(json.dumps({"qa": str(qa), "captions": str(caps)}, indent=2))


def cmd_describe(args: argparse.Namespace) -> None:
    dataset = DenseVideoQADataset(DatasetConfig(args.qa, args.captions, max_items=args.max_items))
    print(json.dumps(describe_dataset(dataset), ensure_ascii=False, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    payload = train_from_config(TrainConfig(qa_path=args.qa, captions_path=args.captions, output_path=args.output,
                                            epochs=args.epochs, lr=args.lr, max_items=args.max_items,
                                            seed=args.seed))
    print(json.dumps({"output": args.output, "weights": payload["weights"],
                      "last": payload["history"][-1]}, ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    dataset = DenseVideoQADataset(DatasetConfig(args.qa, args.captions, max_items=args.max_items))
    preds = load_predictions(args.predictions)
    result = evaluate(dataset, preds)
    groups = evaluate_by_group(dataset.items, preds) if args.groups else None
    print_report(result, groups)
    if args.output:
        save_eval_report(args.output, result, groups)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeVi dense event VideoQA project CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("make-sample", help="Create a tiny runnable dataset.")
    p_sample.add_argument("--output-dir", default="data/sample")
    p_sample.set_defaults(func=cmd_make_sample)

    p_desc = sub.add_parser("describe", help="Describe a QA dataset.")
    p_desc.add_argument("--qa", required=True)
    p_desc.add_argument("--captions", default=None)
    p_desc.add_argument("--max-items", type=int, default=0)
    p_desc.set_defaults(func=cmd_describe)

    p_train = sub.add_parser("train", help="Train the lightweight local ranker.")
    p_train.add_argument("--qa", required=True)
    p_train.add_argument("--captions", default=None)
    p_train.add_argument("--output", default="outputs/ranker.json")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=0.3)
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--max-items", type=int, default=0)
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help="Run DeVi inference.")
    infer_parser = make_infer_parser()
    for action in infer_parser._actions:
        if action.dest == "help":
            continue
        kwargs: Dict[str, Any] = {
            "default": action.default,
            "help": action.help,
            "dest": action.dest,
        }
        if getattr(action, "type", None) is not None:
            kwargs["type"] = action.type
        if getattr(action, "choices", None) is not None:
            kwargs["choices"] = action.choices
        if getattr(action, "nargs", None) is not None:
            kwargs["nargs"] = action.nargs
        option_strings = list(action.option_strings)
        if action.__class__.__name__ == "_StoreTrueAction":
            kwargs.pop("type", None)
            p_infer.add_argument(*option_strings, action="store_true", default=action.default, help=action.help, dest=action.dest)
        elif option_strings:
            p_infer.add_argument(*option_strings, **kwargs)
    p_infer.set_defaults(func=run_inference)

    p_eval = sub.add_parser("eval", help="Evaluate prediction JSONL.")
    p_eval.add_argument("--qa", required=True)
    p_eval.add_argument("--captions", default=None)
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--output", default=None)
    p_eval.add_argument("--max-items", type=int, default=0)
    p_eval.add_argument("--groups", action="store_true")
    p_eval.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

