"""How often does the model choose the near option, split by the data the prompt carries? (saved answers only, no model)

Datatype is the bank (full, units, fill, implied, choice4, ...), the number format the horizon is
written in (digits or written), or the unit word it is rendered in (the units and fill banks
render one duration in several units). One line per value, with the number of prompts and the
near share.

    python -m pipeline.s3_analysis.by_datatype.choice_by_datatype /home/Jared/SPAR/results/horizon --datatype bank
    python -m pipeline.s3_analysis.by_datatype.choice_by_datatype results/horizon --datatype unit --banks units fill
"""

import argparse

from pipeline.s3_analysis.split import load_runs, near_share, split_by

DATATYPES = ("bank", "number_format", "unit")


def summarize(rows, datatype):
    """[(value, n prompts, near share)] by `datatype`."""
    return [(value, len(group), near_share(group))
            for value, group in sorted(split_by(rows, datatype).items(), key=lambda item: str(item[0]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder holding the exp0_<model>_think-<mode>_<bank> run folders")
    parser.add_argument("--datatype", choices=DATATYPES, default="bank")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--mode", choices=("on", "off"), default="on")
    parser.add_argument("--banks", nargs="+", default=["full", "units", "fill", "implied", "choice4", "choice8"])
    args = parser.parse_args()
    rows = load_runs(args.root, args.model, args.mode, args.banks)
    print(f"{args.model} thinking {args.mode}: {len(rows)} answered prompts")
    print(f"{args.datatype:14s} {'n':>5s} {'near share':>11s}")
    for value, n, share in summarize(rows, args.datatype):
        print(f"{str(value):14s} {n:5d} {share:11.2f}")


if __name__ == "__main__":
    main()
