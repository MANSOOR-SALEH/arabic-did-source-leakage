#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_encoder.py
==================
Frozen SSL encoder features for AYDID and SADA_S, so the three-protocol
comparison can be run on learned representations as well as channel features.

Same corpora, same clips, same fold logic as the channel-feature runs -- only
the representation changes. Features are mean+std pooled over time from an
average of three hidden layers (one third, two thirds, final), which is the
standard frozen-probe setup.

Sized for a 6 GB card: fp16, batch 1, gradient-free, and clips longer than
MAX_SEC are centre-cropped. Results cache to npz and the script resumes, so it
can be interrupted and restarted.

Cost on an RTX 3060 6GB, roughly:
    wavlm   ~1.5 h per 10k clips
    xlsr    ~1.5 h per 10k clips

Install:
    pip install torch transformers   (CUDA build of torch)

Usage:
    python -m features.extract_encoder --corpus aydid --encoder wavlm --limit 50
    python -m features.extract_encoder --corpus aydid --encoder wavlm
    python -m features.extract_encoder --corpus sada  --encoder wavlm

Output: features/cache/enc_{corpus}_{encoder}.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AYDID_SOURCE_MANIFEST
from config import AYDID_WAV_ROOTS
from config import SADA_SOURCE_MANIFEST
from config import SADA_WAV_ROOTS
from config import FEATURES_CACHE

CORPORA = {
    "aydid": dict(
        manifest=str(AYDID_SOURCE_MANIFEST),
        roots=[str(r) for r in AYDID_WAV_ROOTS],
        path_col="file_path", name_col="wav_name",
        label="dialect", source="show_name", speaker="speaker_id"),
    "sada": dict(
        manifest=str(SADA_SOURCE_MANIFEST),
        roots=[str(r) for r in SADA_WAV_ROOTS],
        path_col="file_path", name_col=None,
        label="dialect", source="ShowName", speaker=None),
}

MODELS = {
    "wavlm": "microsoft/wavlm-large",
    "xlsr": "facebook/wav2vec2-xls-r-300m",
}

SR = 16000
MAX_SEC = 20.0
FLUSH_EVERY = 500


def find_audio(row, cfg):
    p = str(row.get(cfg["path_col"], "") or "")
    cands = [p] + [os.path.join(r, p) for r in cfg["roots"]]
    if cfg.get("name_col"):
        n = str(row.get(cfg["name_col"], "") or "")
        cands += [os.path.join(r, n) for r in cfg["roots"]]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def load_model(name, device):
    import torch
    from transformers import AutoModel, AutoFeatureExtractor
    mid = MODELS[name]
    fe = AutoFeatureExtractor.from_pretrained(mid)
    m = AutoModel.from_pretrained(
        mid, torch_dtype=torch.float16).to(device).eval()
    return m, fe


def embed(y, model, fe, name, device):
    import torch
    with torch.no_grad():
        inp = fe(y, sampling_rate=SR, return_tensors="pt")
        x = inp.input_values.to(device, torch.float16)
        out = model(x, output_hidden_states=True)
        hs = out.hidden_states
        n = len(hs)
        pick = [hs[n // 3], hs[2 * n // 3], hs[-1]]
        h = torch.stack(pick).mean(0).float()          # (1, T, D)
        v = torch.cat([h.mean(1), h.std(1)], dim=-1)   # (1, 2D)
        return v.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), required=True)
    ap.add_argument("--encoder", choices=list(MODELS), default="wavlm")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import librosa
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("!! no CUDA - this will be extremely slow on CPU.")

    cfg = CORPORA[args.corpus]
    df = pd.read_csv(cfg["manifest"])
    df.columns = [c.strip() for c in df.columns]
    if args.corpus == "sada" and "FileName" in df.columns and "Speaker" in df.columns:
        df["speaker_unit"] = df.FileName.astype(str) + ":" + df.Speaker.astype(str)
        cfg = {**cfg, "speaker": "speaker_unit"}
    if args.limit:
        df = df.groupby(cfg["label"], group_keys=False).head(
            max(1, args.limit // df[cfg["label"]].nunique()))

    out_path = FEATURES_CACHE / f"enc_{args.corpus}_{args.encoder}.npz"
    done = {}
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=True)
        for k, v in zip(z["keys"], z["features"]):
            done[str(k)] = v
        print(f"resuming: {len(done)} clips already embedded")

    model, fe = load_model(args.encoder, device)
    print(f"{args.encoder} on {device}: {len(df)} clips from {args.corpus}")

    keys, feats, meta, missing = [], [], [], 0
    for i, r in enumerate(df.to_dict("records")):
        key = str(r.get(cfg.get("name_col") or cfg["path_col"]))
        row_meta = {"key": key,
                    "label": r.get(cfg["label"]),
                    "source": r.get(cfg["source"]),
                    "speaker": r.get(cfg["speaker"])}
        for e in cfg.get("extra", []):
            row_meta[e] = r.get(e)

        if key in done:
            keys.append(key); feats.append(done[key]); meta.append(row_meta)
            continue

        p = find_audio(r, cfg)
        if p is None:
            missing += 1
            continue
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            if len(y) > int(MAX_SEC * SR):
                mid = len(y) // 2
                half = int(MAX_SEC * SR) // 2
                y = y[mid - half:mid + half]
            if len(y) < SR // 4:
                missing += 1
                continue
            v = embed(y, model, fe, args.encoder, device)
        except Exception as e:
            print(f"  !! {key}: {type(e).__name__}: {e}")
            missing += 1
            continue
        keys.append(key); feats.append(v); meta.append(row_meta)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} ...")
        if (i + 1) % FLUSH_EVERY == 0:
            np.savez_compressed(out_path, keys=np.array(keys),
                                features=np.stack(feats),
                                meta=np.array([str(m) for m in meta]))

    F = np.stack(feats)
    md = pd.DataFrame(meta)
    np.savez_compressed(out_path, keys=np.array(keys), features=F,
                        **{c: md[c].astype(str).values for c in md.columns})
    print(f"\nwrote {out_path}  shape={F.shape}  skipped={missing}")
    print(f"  labels={md.label.nunique()}  sources={md.source.nunique()}  "
          f"speakers={md.speaker.nunique()}")


if __name__ == "__main__":
    main()