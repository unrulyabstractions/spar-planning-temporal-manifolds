"""How often does the model choose the near option, split by a record label? (saved answers only, no model)

One line per value of the label, with the number of prompts and the near share among natural
closes and among forced closes (forced = the reasoning hit the token budget and was cut off).

    python -m pipeline.s3_analysis.by_label.choice_by_label /home/Jared/SPAR/results/horizon --label regime
    python -m pipeline.s3_analysis.by_label.choice_by_label results/horizon --label truth --banks full units
"""

import argparse

from pipeline.s3_analysis.split import load_runs, near_share, split_by

LABELS = ("regime", "unit", "delay", "truth", "n_choices")


def summarize(rows, label):
    """[(value, n prompts, near share of natural closes, near share of forced closes)] by `label`."""
    lines = []
    for value, group in sorted(split_by(rows, label).items(), key=lambda item: str(item[0])):
        natural = [r for r in group if not r["forced_close"]]
        forced = [r for r in group if r["forced_close"]]
        lines.append((value, len(group), near_share(natural), near_share(forced)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder holding the exp0_<model>_think-<mode>_<bank> run folders")
    parser.add_argument("--label", choices=LABELS, default="regime")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--mode", choices=("on", "off"), default="on")
    parser.add_argument("--banks", nargs="+", default=["full", "units", "fill"])
    args = parser.parse_args()
    rows = load_runs(args.root, args.model, args.mode, args.banks)
    print(f"{args.model} thinking {args.mode}, banks {', '.join(args.banks)}: {len(rows)} answered prompts")
    print(f"{args.label:14s} {'n':>5s} {'near | natural':>15s} {'near | forced':>14s}")
    for value, n, natural, forced in summarize(rows, args.label):
        print(f"{str(value):14s} {n:5d} {natural:15.2f} {forced:14.2f}")


if __name__ == "__main__":
    main()
