"""What the reasoning reads after E10's patch before reasoning (07 §12).

The last token of the horizon phrase is its unit word (" years"), so patching
that one position can swap the unit and leave the number: the model may then
read "59 years" with donor "10 days" as "59 days". This script decodes every
reasoning trace and counts which horizon it writes (the recipient's, the
donor's, or recipient number + donor unit), and how often the choice follows
the donor text, split by whether the pair's units differ.

Reads only saved E10 results and the 8B tokenizer. No model.

    HF_HUB_OFFLINE=1 python -m pipeline.s2_steering.analysis.exp10_analysis results/horizon
    -> jared/figures/exp10_reasoning.png, <root>/analysis/exp10.json and .txt
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

FIGURES = Path(__file__).resolve().parents[3] / "figures"
RUN = "exp10_Qwen3-8B_reasoning"
E0_RUN = "exp0_Qwen3-8B_think-on_full"
CONDITIONS = ["recipient", "donor_text", "full_phrase_L16", "full_phrase_L12", "full_last_L16", "full_last_L12",
              "subspace_last_L16", "subspace_last_L12", "random_last_L16", "random_last_L12", "self_last_L16"]


def writes(think, number, unit):
    """Whether the trace writes this duration, also as "59-day" ("1 year" does not match "11 years")."""
    return re.search(rf"(?<![\d.,]){re.escape(number)}[\s-]*{unit}", think) is not None


def load(conditions_dir, name, tokenizer):
    rows = json.loads((conditions_dir / f"{name}.json").read_text())["rows"]
    arrays = np.load(conditions_dir / f"{name}.npz")
    thinks = []
    for i in range(len(rows)):
        ids = arrays["ids"][arrays["offsets"][i]:arrays["offsets"][i + 1]]
        thinks.append(tokenizer.decode(ids[arrays["prompt_len"][i]:arrays["answer_start"][i]]))
    return rows, thinks


def condition_stats(rows, thinks, recipient, donor):
    n = len(rows)
    number_r, unit_r = zip(*(r["recipient_horizon"].rstrip("s").split() for r in rows))
    number_d, unit_d = zip(*(r["donor_horizon"].rstrip("s").split() for r in rows))
    units_differ = np.array([a != b for a, b in zip(unit_r, unit_d)])
    flips = np.array([r["label"] != d["label"] for r, d in zip(recipient[:n], donor[:n])])
    follows_donor = np.array([x["label"] == d["label"] for x, d in zip(rows, donor)])
    changed = np.array([x["label"] != r["label"] for x, r in zip(rows, recipient)])
    writes_recipient = np.array([writes(t, a, b) for t, a, b in zip(thinks, number_r, unit_r)])
    # Recipient number + donor unit is a different horizon only when the units differ, so its rate is over those pairs.
    hybrid = np.array([writes(t, nr, ud) for t, nr, ud in zip(thinks, number_r, unit_d)])
    return {
        "n": n,
        "n_units_differ": int(units_differ.sum()),
        "writes_recipient": float(writes_recipient.mean()),
        "writes_recipient_units_differ": float(writes_recipient[units_differ].mean()),
        "writes_recipient_same_unit": float(writes_recipient[~units_differ].mean()),
        "writes_donor": float(np.mean([writes(t, a, b) for t, a, b in zip(thinks, number_d, unit_d)])),
        "writes_hybrid": float(hybrid[units_differ].mean()),
        "writes_hybrid_count": [int(hybrid[units_differ].sum()), int(units_differ.sum())],
        "mentions_typo": float(np.mean(["typo" in t for t in thinks])),
        "forced": float(np.mean([x["forced"] for x in rows])),
        "follows_donor_flip": [int(follows_donor[flips].sum()), int(flips.sum())],
        "follows_donor_flip_units_differ": [int(follows_donor[flips & units_differ].sum()), int((flips & units_differ).sum())],
        "follows_donor_flip_same_unit": [int(follows_donor[flips & ~units_differ].sum()), int((flips & ~units_differ).sum())],
        "changes_label_no_flip": [int(changed[~flips].sum()), int((~flips).sum())],
    }


def rebatching_changes(root, recipient):
    """Unpatched E10 labels that differ from the same prompt's E0 label (07 §12 batch-noise check): E10 batches
    differently from E0, so this is how much the label moves with no patch at all."""
    e0 = {a["sample_uid"]: a["label"] for a in json.loads((root / E0_RUN / "answers.json").read_text())}
    return [int(sum(r["label"] != e0[r["sample_uid"]] for r in recipient)), len(recipient)]


def plot(stats, path):
    names = [n for n in CONDITIONS if n not in ("recipient", "self_last_L16")]
    labels = [n.replace("_", " ") for n in names]
    x = np.arange(len(names))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6))
    for offset, key, label, color in [(-0.27, "writes_recipient", "recipient's horizon", "#9a9893"),
                                      (0.0, "writes_donor", "donor's horizon", "#2a78d6"),
                                      (0.27, "writes_hybrid", "recipient number + donor unit\n(pairs whose units differ)", "#eb6834")]:
        left.bar(x + offset, [stats[n][key] for n in names], 0.27, label=label, color=color)
    left.set_title("What the reasoning writes")
    left.set_ylabel("share of traces")
    left.legend(fontsize=8)
    for offset, key, label, color in [(-0.2, "follows_donor_flip_units_differ", "units differ", "#eb6834"),
                                      (0.2, "follows_donor_flip_same_unit", "same unit", "#9a9893")]:
        right.bar(x + offset, [stats[n][key][0] / stats[n][key][1] for n in names], 0.4, label=label, color=color)
    right.set_title("Choice follows the donor text\n(pairs where the donor text changes the label)")
    right.set_ylabel("share of pairs")
    right.legend(fontsize=8)
    for axis in (left, right):
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        axis.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    root = Path(sys.argv[1])
    conditions_dir = root / RUN / "conditions"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    loaded = {n: load(conditions_dir, n, tokenizer) for n in CONDITIONS}
    recipient, donor = loaded["recipient"][0], loaded["donor_text"][0]
    stats = {n: condition_stats(rows, thinks, recipient, donor) for n, (rows, thinks) in loaded.items()}
    summary = json.loads((root / RUN / "summary.json").read_text())
    for n in CONDITIONS:
        stats[n]["transfer"] = summary[n.replace("_", " ")]["transfer"]
    stats["criteria"] = summary["criteria"]
    stats["unpatched_label_differs_from_e0"] = rebatching_changes(root, recipient)

    lines = []
    for n in CONDITIONS:
        s = stats[n]
        frac = lambda key: f"{s[key][0]}/{s[key][1]}"
        lines.append(f"{n:18} writes rec {s['writes_recipient']:.2f} "
                     f"(units differ {s['writes_recipient_units_differ']:.2f}, same {s['writes_recipient_same_unit']:.2f}) "
                     f"donor {s['writes_donor']:.2f} hybrid {frac('writes_hybrid_count')} units-differ pairs | follows donor {frac('follows_donor_flip')} "
                     f"(units differ {frac('follows_donor_flip_units_differ')}, same unit {frac('follows_donor_flip_same_unit')}) "
                     f"| changes label w/o flip {frac('changes_label_no_flip')} | transfer {s['transfer'][0]:+.2f} "
                     f"| forced {s['forced']:.2f} typo {s['mentions_typo']:.2f}")
    lines.append("unpatched E10 labels that differ from the E0 label of the same prompt: "
                 "%d/%d" % tuple(stats["unpatched_label_differs_from_e0"]))
    text = "\n".join(lines)
    print(text)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    (out / "exp10.txt").write_text(text + "\n")
    (out / "exp10.json").write_text(json.dumps(stats, indent=1))
    FIGURES.mkdir(exist_ok=True)
    plot(stats, FIGURES / "exp10_reasoning.png")


if __name__ == "__main__":
    main()
