#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_acoustic_sada.py
========================
Channel features for SADA_S, identical definition to the AYDID extractor, so
both corpora are directly comparable.

SADA_S matters because it is the only CROSSED corpus in the set: 63 shows,
82.5% of them carrying more than one dialect. Holding out whole shows removes
recording provenance while every dialect survives in training from other shows.

Speaker unit is the FileName:Speaker composite, not the raw Speaker column.
Duration is SegmentLength, not FullFileLength.

Clips are segments of longer recordings, so audio is loaded with an offset and
duration from SegmentStart/SegmentLength if the per-clip wav does not exist.

Usage:
    python -m features.extract_acoustic_sada --limit 200
    python -m features.extract_acoustic_sada

Output: features/cache/sada_channel_features_{full,speech,nonspeech,cmvn}.csv
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SADA_ROOT as ROOT
from config import SADA_SOURCE_MANIFEST as MANIFEST
from config import FEATURES_CACHE

warnings.filterwarnings("ignore")

SR = 16000
N_FFT, HOP = 512, 160
N_MELS, N_MFCC = 26, 20
PCTL = 30


def find_audio(row):
    fp = str(row.get("file_path", "") or "")
    for c in [fp, os.path.join(ROOT, fp),
              os.path.join(ROOT, "WAVS", fp),
              os.path.join(ROOT, "wavs", fp)]:
        if c and os.path.isfile(c):
            return c
    return None


def frame_stats(S, mfcc, zcr, rms, keep, prefix=""):
    import librosa
    if keep.sum() < 3:
        keep = np.ones(S.shape[1], bool)
    Sk = S[:, keep]
    f = {}
    mel = librosa.feature.melspectrogram(S=Sk ** 2, sr=SR, n_mels=N_MELS)
    ltas = np.log(mel.mean(axis=1) + 1e-10)
    f.update({f"{prefix}ltas{i:02d}": v for i, v in enumerate(ltas - ltas.mean())})
    for name, fn in [("centroid", librosa.feature.spectral_centroid),
                     ("rolloff", librosa.feature.spectral_rolloff),
                     ("bandwidth", librosa.feature.spectral_bandwidth),
                     ("flatness", librosa.feature.spectral_flatness)]:
        v = fn(S=Sk) if name == "flatness" else fn(S=Sk, sr=SR)
        f[f"{prefix}{name}_mean"] = float(np.mean(v))
        f[f"{prefix}{name}_sd"] = float(np.std(v))
    freqs = np.maximum(librosa.fft_frequencies(sr=SR, n_fft=N_FFT), 1.0)
    logp = np.log(Sk.mean(axis=1) + 1e-10)
    f[f"{prefix}tilt"] = float(np.polyfit(np.log(freqs), logp, 1)[0])
    e = rms[keep] if keep.sum() >= 3 else rms
    nf = float(np.percentile(e, 10) + 1e-10)
    f[f"{prefix}noise_floor"] = float(20 * np.log10(nf))
    f[f"{prefix}snr"] = float(20 * np.log10((np.percentile(e, 90) + 1e-10) / nf))
    f[f"{prefix}zcr_mean"] = float(np.mean(zcr[keep]))
    f[f"{prefix}zcr_sd"] = float(np.std(zcr[keep]))
    f[f"{prefix}rms_mean"] = float(np.mean(e))
    f[f"{prefix}rms_sd"] = float(np.std(e))
    mk = mfcc[:, keep]
    f.update({f"{prefix}mfcc{i:02d}": float(v) for i, v in enumerate(mk.mean(axis=1))})
    return f


def extract_one(path, offset=None, duration=None):
    import librosa
    kw = {}
    if offset is not None and duration is not None and duration > 0:
        kw = dict(offset=float(offset), duration=float(duration))
    y, _ = librosa.load(path, sr=SR, mono=True, **kw)
    if len(y) < N_FFT * 2:
        return None
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    n = min(S.shape[1], len(rms), len(zcr))
    S, rms, zcr = S[:, :n], rms[:n], zcr[:n]
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS)), n_mfcc=N_MFCC)
    lo, hi = np.percentile(rms, PCTL), np.percentile(rms, 100 - PCTL)
    out = {
        "full": frame_stats(S, mfcc, zcr, rms, np.ones(n, bool)),
        "speech": frame_stats(S, mfcc, zcr, rms, rms >= hi),
        "nonspeech": frame_stats(S, mfcc, zcr, rms, rms <= lo),
    }
    logS = np.log(S + 1e-10)
    Sc = np.exp((logS - logS.mean(axis=1, keepdims=True)) /
                (logS.std(axis=1, keepdims=True) + 1e-10))
    mfcc_c = mfcc - mfcc.mean(axis=1, keepdims=True)
    out["cmvn"] = frame_stats(Sc, mfcc_c, zcr, rms, np.ones(n, bool))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # composite speaker unit
    if "FileName" in df.columns and "Speaker" in df.columns:
        df["speaker_unit"] = (df["FileName"].astype(str) + ":" +
                              df["Speaker"].astype(str))
    else:
        df["speaker_unit"] = df.get("speaker_id", pd.Series(range(len(df)))).astype(str)

    if args.limit:
        df = df.groupby("dialect", group_keys=False).head(
            max(1, args.limit // max(df.dialect.nunique(), 1)))
    print(f"extracting: {len(df)} clips")

    rows = {k: [] for k in ["full", "speech", "nonspeech", "cmvn"]}
    missing, fails = 0, []
    for i, r in enumerate(df.to_dict("records")):
        p = find_audio(r)
        if p is None:
            missing += 1
            fails.append((r.get("ShowName"), r.get("file_path"), "missing"))
            continue
        try:
            feats = extract_one(p)
        except Exception as e:
            missing += 1
            fails.append((r.get("ShowName"), r.get("file_path"), type(e).__name__))
            continue
        if feats is None:
            missing += 1
            fails.append((r.get("ShowName"), r.get("file_path"), "too short"))
            continue
        meta = {"clip_id": r.get("SegmentID", r.get("file_path")),
                "speaker": r.get("speaker_unit"),
                "dialect": r.get("dialect"),
                "show_name": r.get("ShowName"),
                "environment": r.get("Environment"),
                "duration": r.get("SegmentLength")}
        for k, f in feats.items():
            rows[k].append({**meta, **f})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df)} ...")

    META = ("clip_id", "speaker", "dialect", "show_name", "environment", "duration")
    for k, v in rows.items():
        out = pd.DataFrame(v)
        name = FEATURES_CACHE / f"sada_channel_features_{k}.csv"
        out.to_csv(name, index=False)
        print(f"  wrote {name}  ({len(out)} rows, "
              f"{len([c for c in out.columns if c not in META])} features)")
    print(f"  skipped/unreadable: {missing}")
    if fails:
        fd = pd.DataFrame(fails, columns=["show", "path", "reason"])
        fd.to_csv(FEATURES_CACHE / "sada_extraction_failures.csv", index=False)
        print(fd.groupby("reason").size().to_string())
        if missing > len(df) * 0.5:
            print("  >>> most files not found — check the audio layout under")
            print(f"      {ROOT} and adjust find_audio().")


if __name__ == "__main__":
    main()