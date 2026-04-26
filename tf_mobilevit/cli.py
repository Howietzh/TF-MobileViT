from __future__ import annotations

import argparse
import json

from .ablations import run_ablations
from .config import load_config
from .engine import evaluate, infer_image, train


def apply_dataset_overrides(config: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "train_split_file", None):
        config["dataset"]["train_split_file"] = args.train_split_file
    if getattr(args, "val_split_file", None):
        config["dataset"]["val_split_file"] = args.val_split_file
    if getattr(args, "test_split_file", None):
        config["dataset"]["test_split_file"] = args.test_split_file
    return config


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-split-file", help="Training split txt file, e.g. train_subset_3.txt")
    parser.add_argument("--val-split-file", help="Validation split txt file")
    parser.add_argument("--test-split-file", help="Test split txt file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tf-mobilevit")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", default="configs/default.yaml")
    add_split_args(train_p)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--config", default="configs/default.yaml")
    eval_p.add_argument("--checkpoint", required=True)
    add_split_args(eval_p)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--config", default="configs/default.yaml")
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--input", required=True)
    infer_p.add_argument("--output", required=True)

    ablate_p = sub.add_parser("ablate")
    ablate_p.add_argument("--config", default="configs/default.yaml")
    add_split_args(ablate_p)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_dataset_overrides(load_config(args.config), args)
    if args.command == "train":
        result = train(config)
        print(json.dumps(result, indent=2))
    elif args.command == "eval":
        result = evaluate(config, args.checkpoint)
        print(json.dumps(result, indent=2))
    elif args.command == "infer":
        out = infer_image(config, args.checkpoint, args.input, args.output)
        print(out)
    elif args.command == "ablate":
        result = run_ablations(config)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
