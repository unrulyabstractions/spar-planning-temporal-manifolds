"""E8: causal tests of the horizon code -- patching, ablation and steering in the
residual stream (spec: 07_spec_steering.md).

Layer convention: `layer` is a hidden_states index, as in the saved activations
(L0 = embeddings, L{i} = output of decoder layer i - 1), so an edit at layer i
hooks model.model.layers[i - 1]. The last index is excluded: transformers records
it after the final norm, so it is not the last decoder layer's raw output.

Outcome: the near-option log-odds, logit(near label) - logit(far label), read
at the last position of each sequence in one forward pass.
"""

from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .full_bank import HORIZONS, regime, render_variant
from .geometry import _fit
from .model_io import chat_text, suffix_length


# --- edits: called with the selected activations x [n, d] (float32), the batch row
# of each, and its slot in the batch's position index; they return the new values.

class Add:
    """Add a per-row vector: `delta` [batch, d]."""

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, x, rows, slots):
        return x + self.delta[rows]


class MeanAblate:
    """Set the component of x in the span of `basis` [k, d] (orthonormal rows) to
    that of `mean` [d]: x - ((x - mean) B^T) B."""

    def __init__(self, mean, basis):
        self.mean, self.basis = mean, basis

    def __call__(self, x, rows, slots):
        return x - ((x - self.mean) @ self.basis.T) @ self.basis


class Replace:
    """Overwrite with donor values, given in the same order as the selection."""

    def __init__(self, values):
        self.values = values

    def __call__(self, x, rows, slots):
        return self.values


@contextmanager
def edit_residual(model, layer, index, mask, edit):
    """Within the block, decoder layer `layer - 1` passes its output through
    `edit` at the positions index[b, p] where mask[b, p] (padded coordinates)."""
    if not 1 <= layer < len(model.model.layers):
        raise ValueError(f"layer {layer}: editable hidden_states indices are 1..{len(model.model.layers) - 1} "
                         "(0 is the embeddings, the last is recorded after the final norm)")
    batch, width = index.shape
    rows = torch.arange(batch, device=index.device)[:, None].expand(batch, width)[mask]
    slots = torch.arange(width, device=index.device)[None, :].expand(batch, width)[mask]
    positions = index[mask]

    def hook(_module, _inputs, out):
        h = (out[0] if isinstance(out, tuple) else out).clone()
        h[rows, positions] = edit(h[rows, positions].float(), rows, slots).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --- batched scoring ----------------------------------------------------------

def pad_left(id_lists, pad_id):
    """Left-padded ids [B, T], attention mask, and each row's offset."""
    width = max(len(ids) for ids in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(id_lists), width), dtype=torch.long)
    offsets = []
    for row, seq in enumerate(id_lists):
        offset = width - len(seq)
        ids[row, offset:] = torch.tensor(seq, dtype=torch.long)
        attention[row, offset:] = 1
        offsets.append(offset)
    return ids, attention, offsets


def place(positions, offsets):
    """Per-row unpadded positions -> padded index [B, P] and mask [B, P]."""
    width = max(1, max(len(p) for p in positions))
    index = torch.zeros((len(positions), width), dtype=torch.long)
    mask = torch.zeros((len(positions), width), dtype=torch.bool)
    for row, (pos, offset) in enumerate(zip(positions, offsets)):
        index[row, :len(pos)] = torch.tensor(pos, dtype=torch.long) + offset
        mask[row, :len(pos)] = True
    return index, mask


@torch.no_grad()
def forward(model, device, id_lists, pad_id, hidden=False, layer=None, positions=None, edit=None):
    """Last-position logits [B, vocab] (float32), the hidden states if asked, and
    the row offsets. Position ids follow the attention mask, so left padding
    leaves each row's computation as if it were alone. The recorded hidden state
    at an edited layer is the unedited one (transformers records it before
    forward hooks run); the edit shows from the next layer on."""
    ids, attention, offsets = pad_left(id_lists, pad_id)
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    context = nullcontext()
    if edit is not None:
        index, mask = place(positions, offsets)
        context = edit_residual(model, layer, index.to(device), mask.to(device), edit)
    with context:
        # only the last position's logits are used; computing all of them costs a
        # [B, T, vocab] tensor, about 7 GB for 8 cached reasoning traces
        out = model(input_ids=ids.to(device), attention_mask=attention.to(device),
                    position_ids=position_ids.to(device), output_hidden_states=hidden, logits_to_keep=1)
    return out.logits[:, -1].float(), (out.hidden_states if hidden else None), offsets


@torch.no_grad()
def forward_capture_last(model, device, id_lists, pad_id):
    """Last-position logits [B, vocab] and each editable layer's output at the last
    position: a list where entry L - 1 is hidden_states index L, [B, d] float32,
    for L = 1 .. n_layers - 1. Only the last position is kept, so long cached
    reasoning traces fit (all hidden states of 8 traces would be ~7 GB)."""
    captured = []
    blocks = model.model.layers[:-1]

    def hook(_module, _inputs, out):
        captured.append((out[0] if isinstance(out, tuple) else out)[:, -1].float())

    handles = [block.register_forward_hook(hook) for block in blocks]
    try:
        logits, _, _ = forward(model, device, id_lists, pad_id)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def near_log_odds(logits, near_first, labels):
    """logit(near label) - logit(far label) per row; labels[row] = (id of a, id of b)
    in the form the model emits there (with or without a leading space)."""
    labels = torch.as_tensor(np.asarray(labels), device=logits.device)
    rows = torch.arange(len(labels), device=logits.device)
    diff = (logits[rows, labels[:, 0]] - logits[rows, labels[:, 1]]).cpu().numpy()
    return np.where(np.asarray(near_first), diff, -diff)


def score(model, device, id_lists, near_first, labels, pad_id, batch_size,
          layer=None, positions=None, make_edit=None):
    """Near log-odds for every sequence, in batches. `labels[row]` = (a, b) token
    ids. `make_edit(rows)` builds the edit for one batch (rows index into
    `id_lists`), applied at `layer` on `positions[row]` (unpadded)."""
    out = []
    for start in range(0, len(id_lists), batch_size):
        rows = list(range(start, min(start + batch_size, len(id_lists))))
        edit = make_edit(rows) if make_edit is not None else None
        logits, _, _ = forward(model, device, [id_lists[r] for r in rows], pad_id, layer=layer,
                               positions=[positions[r] for r in rows] if edit is not None else None, edit=edit)
        out.append(near_log_odds(logits, [near_first[r] for r in rows], [labels[r] for r in rows]))
    return np.concatenate(out)


@torch.no_grad()
def score_last_edits(model, device, id_lists, near_first, labels, pad_id, batch_size, edits):
    """Near log-odds [len(edits), n] for edits at the last position only. `edits`
    holds (layer, make_edit) pairs, or None for the unedited sequences. A causal
    model's states before the last position do not depend on an edit there, so
    each batch's prefix (all but the last token) runs once and its key/value cache
    serves every edit, which then costs a one-token pass. Equals score() per edit;
    on ~2.5k-token reasoning traces this is the difference between hours and minutes.
    Models with recurrent layers (linear attention) fall back to full passes: their
    cache holds a running state that cannot be cropped back to the prefix."""
    layer_types = getattr(model.config.get_text_config(), "layer_types", None) or ["full_attention"]
    if any(t != "full_attention" for t in layer_types):
        last = [[len(seq) - 1] for seq in id_lists]
        return np.stack([score(model, device, id_lists, near_first, labels, pad_id, batch_size)
                         if spec is None else
                         score(model, device, id_lists, near_first, labels, pad_id, batch_size, spec[0], last, spec[1])
                         for spec in edits])
    out = np.zeros((len(edits), len(id_lists)))
    one = torch.ones((1, 1), dtype=torch.bool, device=device)
    for start in range(0, len(id_lists), batch_size):
        rows = list(range(start, min(start + batch_size, len(id_lists))))
        ids, attention, _ = pad_left([id_lists[r] for r in rows], pad_id)
        if ids.shape[1] < 2:
            raise ValueError("score_last_edits needs at least two tokens per sequence")
        ids, attention = ids.to(device), attention.to(device)
        position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
        cache = model(input_ids=ids[:, :-1], attention_mask=attention[:, :-1], position_ids=position_ids[:, :-1],
                      use_cache=True, logits_to_keep=1).past_key_values
        # left padding puts every row's last token in the last column: slot 0 of the one-token pass
        index, mask = torch.zeros((len(rows), 1), dtype=torch.long, device=device), one.expand(len(rows), 1)
        for e, spec in enumerate(edits):
            context = nullcontext() if spec is None else edit_residual(model, spec[0], index, mask, spec[1](rows))
            with context:
                logits = model(input_ids=ids[:, -1:], attention_mask=attention, position_ids=position_ids[:, -1:],
                               past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
            # drop the edited last token, so the next edit sees only the prefix; a negative count,
            # since newer transformers reject crop() to a positive length
            cache.crop(-1)
            out[e, start:start + len(rows)] = near_log_odds(logits, [near_first[r] for r in rows],
                                                            [labels[r] for r in rows])
    return out


def gather(hidden, positions, offsets):
    """Values of one layer's hidden states [B, T, d] at per-row unpadded positions,
    flattened in the order edit_residual selects them (for Replace)."""
    index, mask = place(positions, offsets)
    index, mask = index.to(hidden.device), mask.to(hidden.device)
    rows = torch.arange(index.shape[0], device=hidden.device)[:, None].expand_as(index)[mask]
    return hidden[rows, index[mask]].float()


# --- directions from saved activations ----------------------------------------

def fit_cell(X, log_h, k_max=8):
    """Everything E8 needs at one cell, from activations X [n, d] and log10
    horizons: the mean, the ridge probe (w, b) for log horizon, the top-k PCs of
    the per-horizon averages with each average's coordinates in them (the
    curve), and the long-minus-short mean difference."""
    mean = X.mean(0)
    Xc = X - mean
    probe = _fit(Xc, log_h)
    groups = np.round(log_h, 9)
    horizons = np.unique(groups)
    averages = np.stack([Xc[groups == h].mean(0) for h in horizons])
    k = min(k_max, len(horizons) - 1)
    basis = PCA(n_components=k).fit(averages).components_
    median = np.median(log_h)
    return {"mean": mean, "w": probe.coef_.astype(np.float64), "b": float(probe.intercept_),
            "basis": basis, "curve_logh": horizons, "curve_coords": averages @ basis.T,
            "caa": X[log_h >= median].mean(0) - X[log_h < median].mean(0)}


def curve_delta(cell, log_h, alpha):
    """Per-row displacement [n, d] along the curve of horizon averages, from each
    row's horizon to that horizon + alpha decades (clamped at the curve's ends)."""
    grid, coords = cell["curve_logh"], cell["curve_coords"]

    def at(points):
        return np.stack([np.interp(points, grid, coords[:, j]) for j in range(coords.shape[1])], 1)

    return (at(log_h + alpha) - at(log_h)) @ cell["basis"]


def matched_deltas(cell, vector, alpha, log_h):
    """Per-row displacement along `vector`, each row as long as its displacement
    along the curve for the same alpha, so straight and curved moves differ in
    direction only. (A probe-readout-matched step would be 10-19x shorter than
    the real spacing of horizons on the 8B: the probe weights fast-changing,
    low-variance directions.)"""
    lengths = np.linalg.norm(curve_delta(cell, log_h, alpha), axis=1, keepdims=True)
    return np.sign(alpha) * lengths * vector / np.linalg.norm(vector)


def decision_direction(X, near_lo, cell, folds=5):
    """The decision inside the cell's horizon subspace: a ridge from the coordinates on
    the horizon-average basis to the near log-odds, mapped back to a unit vector [d]
    that raises the near log-odds. Kept inside the subspace so the curve can be split
    into its decision part and the rest. Returns (vector, out-of-fold R²)."""
    Z = (X - cell["mean"]) @ cell["basis"].T
    pred = np.empty(len(near_lo))
    for train, test in KFold(folds, shuffle=True, random_state=0).split(Z):
        pred[test] = _fit(Z[train], near_lo[train]).predict(Z[test])
    r2 = 1 - ((near_lo - pred) ** 2).sum() / ((near_lo - near_lo.mean()) ** 2).sum()
    vector = _fit(Z, near_lo).coef_ @ cell["basis"]
    return vector / np.linalg.norm(vector), float(r2)


def without_direction(deltas, unit):
    """Rows of `deltas` [n, d] with their component along the unit vector removed."""
    return deltas - np.outer(deltas @ unit, unit)


def basis_without(basis, unit):
    """Orthonormal basis [k - 1, d] of the part of span(basis) [k, d] orthogonal to
    `unit`'s projection onto it: the horizon subspace with the decision taken out."""
    coords = basis @ unit
    coords = coords / np.linalg.norm(coords)
    # QR of [coords | identity]: the first column spans coords, the next k - 1 complete it
    q, _ = np.linalg.qr(np.c_[coords, np.eye(len(coords))])
    return q[:, 1:len(coords)].T @ basis


def random_orthonormal(d, k, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T


# --- prompts -------------------------------------------------------------------

def encode(tokenizer, prompt, cfg):
    return tokenizer.encode(chat_text(tokenizer, prompt, cfg), add_special_tokens=False)


def crossing_pairs(records, tokenizer, cfg, rng, limit):
    """For each near- or both-regime record, the same prompt with a horizon from
    the other of those two regimes and the same token length. Returns dicts
    with recipient/donor ids, the record index, the donor horizon, the span
    (start, end) of the horizon phrase in both prompts and the prompt end (first template-suffix token)."""
    n_suffix = suffix_length(tokenizer, records[0]["prompt"], cfg)
    pairs = []
    for i, rec in enumerate(records):
        here = regime(rec["horizon"], rec["near_delay_years"], rec["far_delay_years"])
        if here not in ("near", "both"):
            continue
        other = "both" if here == "near" else "near"
        recipient = encode(tokenizer, rec["prompt"], cfg)
        recipient_phrase = phrase_positions(tokenizer, chat_text(tokenizer, rec["prompt"], cfg), rec["horizon_text"])
        options = [h for h in HORIZONS if regime(h.years, rec["near_delay_years"], rec["far_delay_years"]) == other]
        rng.shuffle(options)
        for h in options:
            donor_prompt = render_variant({**rec, "horizon_text": str(h)})
            donor = encode(tokenizer, donor_prompt, cfg)
            if len(donor) != len(recipient):
                continue
            # the whole phrase, not just the tokens that differ: "91 years" -> "16 years" shares " years"
            phrase = recipient_phrase + phrase_positions(tokenizer, chat_text(tokenizer, donor_prompt, cfg), str(h))
            pairs.append({"recipient": recipient, "donor": donor, "record": i, "donor_horizon": str(h),
                          "span": (min(phrase), max(phrase) + 1), "prompt_end": len(recipient) - n_suffix})
            break
        if len(pairs) >= limit:
            break
    return pairs


def answer_pairs(regimes, near_labels, rng, limit):
    """(recipient, donor) index pairs for answer-slot patching across prompts: the
    donor is in the other of the near/both regimes and has its near option under
    the same label, so "prefer a)" means the same option in both."""
    rows = list(range(len(regimes)))
    rng.shuffle(rows)
    pairs = []
    for r in rows:
        other = "both" if regimes[r] == "near" else "near"
        donors = [d for d in range(len(regimes)) if regimes[d] == other and near_labels[d] == near_labels[r]]
        if donors:
            pairs.append((r, rng.choice(donors)))
        if len(pairs) >= limit:
            break
    return pairs


def patch_groups(pair, tokenizer):
    """(name, positions) for T1: the horizon span, the prompt after it, then each
    template-suffix token named by its text."""
    start, end = pair["span"]
    groups = [("horizon", list(range(start, end))),
              ("prompt after horizon", list(range(end, pair["prompt_end"])))]
    for p in range(pair["prompt_end"], len(pair["recipient"])):
        groups.append((repr(tokenizer.decode([pair["recipient"][p]])), [p]))
    return groups


# --- statistics ----------------------------------------------------------------

def toward_rational(effect, regimes, alpha):
    """Per-prompt effect signed toward the answer the target horizon makes rational,
    NaN where the target stays in the same regime: a longer horizon (alpha > 0)
    turns near-regime prompts toward far (negative effect), a shorter one
    (alpha < 0) turns both-regime prompts toward near (positive effect)."""
    regimes = np.asarray(regimes)
    out = np.full(len(effect), np.nan)
    if alpha > 0:
        out[regimes == "near"] = -effect[regimes == "near"]
    else:
        out[regimes == "both"] = effect[regimes == "both"]
    return out


def removed_fraction(base, edited, regimes, n=1000, seed=0):
    """(mean, low, high): the share of the near-minus-both contrast an edit removes,
    1 - contrast(edited) / contrast(base), bootstrapping prompts within each regime."""
    regimes = np.asarray(regimes)
    near, both = np.flatnonzero(regimes == "near"), np.flatnonzero(regimes == "both")
    if len(near) == 0 or len(both) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)

    def frac(i_near, i_both):
        c0 = base[i_near].mean() - base[i_both].mean()
        c1 = edited[i_near].mean() - edited[i_both].mean()
        return 1 - c1 / c0

    draws = [frac(rng.choice(near, len(near)), rng.choice(both, len(both))) for _ in range(n)]
    return float(frac(near, both)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_mean(values, n=1000, seed=0):
    """(mean, low, high) with a 95% percentile interval over resampled rows."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrast(lo, regimes):
    """Mean near log-odds in the near regime minus in the both regime (NaN if
    either regime is absent, as in a tiny smoke subset)."""
    regimes = np.asarray(regimes)
    if not ((regimes == "near").any() and (regimes == "both").any()):
        return float("nan")
    return float(lo[regimes == "near"].mean() - lo[regimes == "both"].mean())


# --- E10: patching prompt positions while the model generates -------------------

@contextmanager
def prompt_patch(model, layer, positions, donor, basis=None):
    """Within the block, every forward pass that processes a whole sequence (a
    prefill, not a one-token decode step) has the output of decoder layer
    `layer - 1` overwritten at the unpadded prompt positions `positions[row]` with
    `donor[row]` [len(positions[row]), d]; with `basis` [k, d] (orthonormal rows)
    only the component in its span is swapped: x + ((donor - x) B^T) B. Rows are
    the batch rows in order, so one batch per block. Each row's left padding is
    read from the attention mask of the pass, so the same block covers the
    reasoning pass and the answer pass of `model_io._generate_rows`, whose
    sequences have different lengths."""
    if not 1 <= layer < len(model.model.layers):
        raise ValueError(f"layer {layer}: editable hidden_states indices are 1..{len(model.model.layers) - 1}")
    padding = {}

    def read_padding(_module, _args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.dim() != 2:
            raise ValueError("prompt_patch needs the 2-D attention mask to locate each row's prompt")
        padding["left"] = (mask == 0).sum(1).tolist()

    def patch(_module, _inputs, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:   # a decode step: generated tokens are never patched
            return out
        if h.shape[0] != len(positions):
            raise ValueError(f"batch of {h.shape[0]} rows, patch prepared for {len(positions)}")
        h = h.clone()
        for row, (pos, values) in enumerate(zip(positions, donor)):
            index = torch.as_tensor(pos, device=h.device) + padding["left"][row]
            x = h[row, index].float()
            new = values.to(h.device, torch.float32)
            if basis is not None:
                b = basis.to(h.device, torch.float32)
                new = x + ((new - x) @ b.T) @ b
            h[row, index] = new.to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    handles = [model.model.register_forward_pre_hook(read_padding, with_kwargs=True),
               model.model.layers[layer - 1].register_forward_hook(patch)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def phrase_positions(tokenizer, text, horizon_text, marker="horizon: "):
    """Token indices of the horizon phrase in the chat-templated `text`: the tokens
    overlapping "<horizon_text>" right after `marker` (the bank's constraint line
    reads "... for this time horizon: 3 weeks.")."""
    start = text.index(marker + horizon_text) + len(marker)
    end = start + len(horizon_text)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
