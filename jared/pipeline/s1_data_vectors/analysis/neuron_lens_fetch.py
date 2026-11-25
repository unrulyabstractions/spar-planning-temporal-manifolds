"""Fetch what `neuron_lens_compare` needs (spec 32): the Neuron Lens labels of a few
Qwen3-8B blocks and those blocks' MLP tensors, without downloading the 16 GB checkpoint.

Tensors are read straight out of the Hugging Face safetensors shards with HTTP range
requests (a shard is an 8-byte header length, a JSON header of byte offsets, then raw
data) and saved as raw bf16 bits in uint16 .npy files; `neuron_lens_compare.load_bf16`
turns them into float32. Labels are converted from parquet to JSON so the analysis
needs no parquet reader. Both sources are pinned to the commits in MODEL_SHA and
CATALOG_SHA, recorded in meta.json.

    uv run --no-project --with pyarrow --with numpy --with tqdm \
        python -m pipeline.s1_data_vectors.analysis.neuron_lens_fetch $RESULTS
    -> <root>/neuron_lens/labels_L<b>.json, L<b>_<tensor>.npy, meta.json
"""

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

MODEL_SHA = "b968826d9c46dd6066d109eabc6255188de91218"     # Qwen/Qwen3-8B, 2026-11-12
CATALOG_SHA = "21666b9f217328caee73d10de85ca19f4c174e72"   # stdlogits/neuron-lens, 2026-11-12
MODEL_URL = f"https://huggingface.co/Qwen/Qwen3-8B/resolve/{MODEL_SHA}"
CATALOG_URL = f"https://huggingface.co/datasets/stdlogits/neuron-lens/resolve/{CATALOG_SHA}/qwen3-8b/neurons/labels"

READ_BLOCKS = (16, 22, 24)    # neurons that read the cells' residual (hidden_states[L] enters block L)
WRITE_BLOCKS = (15, 21, 23)   # neurons whose writes land in it
READ_TENSORS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "post_attention_layernorm.weight")
WRITE_TENSORS = ("mlp.down_proj.weight",)
CHUNK = 8 << 20


def get(url, start=None, end=None):
    """The body of `url`, or bytes start..end inclusive."""
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request) as response:
        return response.read()


def shard_header(url):
    """(header dict, offset of the data section) of a safetensors shard."""
    (length,) = struct.unpack("<Q", get(url, 0, 7))
    return json.loads(get(url, 8, 7 + length)), 8 + length


def fetch_tensor(url, entry, data_start, path, progress):
    """Range-read one bf16 tensor into `path` as uint16 bits."""
    if entry["dtype"] != "BF16":
        raise ValueError(f"expected BF16, got {entry['dtype']}")
    begin, stop = entry["data_offsets"]
    out = bytearray()
    for offset in range(begin, stop, CHUNK):
        out += get(url, data_start + offset, data_start + min(offset + CHUNK, stop) - 1)
        progress.update(min(CHUNK, stop - offset))
    if len(out) != stop - begin:
        raise IOError(f"{path.name}: got {len(out)} bytes, expected {stop - begin}")
    np.save(path, np.frombuffer(bytes(out), dtype=np.uint16).reshape(entry["shape"]))


def wanted_tensors():
    """[(block, tensor name)] to fetch."""
    return ([(b, t) for b in READ_BLOCKS for t in READ_TENSORS]
            + [(b, t) for b in WRITE_BLOCKS for t in WRITE_TENSORS])


def main(root):
    out = Path(root) / "neuron_lens"
    out.mkdir(parents=True, exist_ok=True)

    for block in sorted(set(READ_BLOCKS + WRITE_BLOCKS)):
        path = out / f"labels_L{block}.json"
        if not path.exists():
            raw = out / f"labels_L{block:02d}.parquet"
            raw.write_bytes(get(f"{CATALOG_URL}/L{block:02d}.parquet"))
            path.write_text(json.dumps(pq.read_table(raw).to_pylist()))
            raw.unlink()

    weight_map = json.loads(get(f"{MODEL_URL}/model.safetensors.index.json"))["weight_map"]
    headers, plan = {}, []
    for block, tensor in wanted_tensors():
        path = out / f"L{block}_{tensor.replace('.weight', '').replace('.', '_')}.npy"
        if path.exists():
            continue
        name = f"model.layers.{block}.{tensor}"
        shard_url = f"{MODEL_URL}/{weight_map[name]}"
        if shard_url not in headers:
            headers[shard_url] = shard_header(shard_url)
        header, data_start = headers[shard_url]
        plan.append((shard_url, header[name], data_start, path))

    total = sum(e["data_offsets"][1] - e["data_offsets"][0] for _, e, _, _ in plan)
    with tqdm(total=total, unit="B", unit_scale=True, desc="tensors") as progress:
        for shard_url, entry, data_start, path in plan:
            fetch_tensor(shard_url, entry, data_start, path, progress)

    (out / "meta.json").write_text(json.dumps({
        "model": "Qwen/Qwen3-8B", "model_sha": MODEL_SHA,
        "catalog": "stdlogits/neuron-lens", "catalog_sha": CATALOG_SHA,
        "read_blocks": READ_BLOCKS, "write_blocks": WRITE_BLOCKS}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="results root, e.g. /home/Jared/SPAR/results/horizon")
    main(parser.parse_args().root)
