#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_acoustic.py
===================
Extracts the non-temporal ("channel-only") feature set for AYDID, in the four
variants the paper's argument depends on:

    full        all frames
    speech      top-30% energy frames only
    nonspeech   bottom-30% energy frames only   <- the key control: no phonemes
    cmvn        full, after per-utterance mean/variance normalisation

Every feature is averaged over the utterance, so no temporal phonetic sequence
survives. No neural encoder is involved.

Feature set (matches the SADIC extraction):
    LTAS shape (log-mel band means), spectral centroid / rolloff / bandwidth /
    flatness (mean+sd), spectral tilt, noise floor, SNR estimate, zero-crossing
    rate (mean+sd), RMS (mean+sd), MFCC means (20).

Frame selection uses an energy percentile computed WITHIN each utterance, so it
is independent of how loud the recording is overall.

~17.5k files. Roughly 20-40 minutes on CPU; use --limit to smoke-test first.

Usage:
    python -m features.extract_acoustic --limit 200     # quick check
    python -m features.extract_acoustic                 # full run

Output: aydid_channel_features_{full,speech,nonspeech,cmvn}.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AYDID_ROOT as ROOT
from config import AYDID_SOURCE_MANIFEST as MANIFEST
from config import FEATURES_CACHE

DUR_CACHE = FEATURES_CACHE / "aydid_durations.csv"
MAX_DUR = 30.0
SR = 16000
N_FFT, HOP = 512, 160
N_MELS, N_MFCC = 26, 20
PCTL = 30            # bottom/top 30% of frames by energy


def resolve_wav_root():
    for c in [os.path.join(ROOT, "WAVS_16K"), os.path.join(ROOT, "WAVS")]:
        if os.path.isdir(c):
            return c
    raise SystemExit("no WAVS_16K or WAVS directory found")


def find_audio(row, wav_root):
    fp = str(row.get("file_path", "") or "")
    for c in [fp, os.path.join(wav_root, fp),
              os.path.join(wav_root, str(row.get("wav_name", "") or "")),
              os.path.join(wav_root, str(row.get("dialect", "")),
                           str(row.get("speaker_id", "")),
                           str(row.get("wav_name", "") or ""))]:
        if c and os.path.isfile(c):
            return c
    return None


def frame_stats(S, mfcc, zcr, rms, keep, prefix=""):
    """Aggregate the retained frames into utterance-level statistics."""
    import librosa
    if keep.sum() < 3:
        keep = np.ones(S.shape[1], bool)
    Sk = S[:, keep]
    f = {}
    # LTAS shape: log-mel band means, normalised to remove overall gain
    mel = librosa.feature.melspectrogram(S=Sk ** 2, sr=SR, n_mels=N_MELS)
    ltas = np.log(mel.mean(axis=1) + 1e-10)
    f.update({f"{prefix}ltas{i:02d}": v for i, v in enumerate(ltas - ltas.mean())})
    # spectral shape descriptors
    for name, fn in [("centroid", librosa.feature.spectral_centroid),
                     ("rolloff", librosa.feature.spectral_rolloff),
                     ("bandwidth", librosa.feature.spectral_bandwidth),
                     ("flatness", librosa.feature.spectral_flatness)]:
        v = fn(S=Sk) if name == "flatness" else fn(S=Sk, sr=SR)
        f[f"{prefix}{name}_mean"] = float(np.mean(v))
        f[f"{prefix}{name}_sd"] = float(np.std(v))
    # spectral tilt: slope of log power against log frequency
    freqs = np.maximum(librosa.fft_frequencies(sr=SR, n_fft=N_FFT), 1.0)
    logp = np.log(Sk.mean(axis=1) + 1e-10)
    f[f"{prefix}tilt"] = float(np.polyfit(np.log(freqs), logp, 1)[0])
    # noise floor and a crude SNR, from the frame energy distribution
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


def extract_one(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
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
    # CMVN: per-utterance log-spectral mean subtraction + variance normalisation
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

    wav_root = resolve_wav_root()
    df = pd.read_csv(MANIFEST)
    if os.path.exists(DUR_CACHE):
        df = df.merge(pd.read_csv(DUR_CACHE), on="wav_name", how="left")
        df = df[df["duration"].isna() | (df["duration"] <= MAX_DUR)]
    if args.limit:
        df = df.groupby("dialect", group_keys=False).head(
            max(1, args.limit // df.dialect.nunique()))
    print(f"extracting from {wav_root}: {len(df)} clips")

    rows = {k: [] for k in ["full", "speech", "nonspeech", "cmvn"]}
    missing = 0
    for i, r in enumerate(df.to_dict("records")):
        p = find_audio(r, wav_root)
        if p is None:
            missing += 1
            continue
        try:
            feats = extract_one(p)
        except Exception as e:
            print(f"  !! {r.get('wav_name')}: {type(e).__name__}: {e}")
            missing += 1
            continue
        if feats is None:
            missing += 1
            continue
        meta = {"wav_name": r["wav_name"], "speaker": r["speaker_id"],
                "dialect": r["dialect"], "show_name": r["show_name"],
                "duration": r.get("duration", np.nan)}
        for k, f in feats.items():
            rows[k].append({**meta, **f})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df)} ...")

    for k, v in rows.items():
        out = pd.DataFrame(v)
        name = FEATURES_CACHE / f"aydid_channel_features_{k}.csv"
        out.to_csv(name, index=False)
        nfeat = len([c for c in out.columns if c not in
                     ("wav_name", "speaker", "dialect", "show_name", "duration")])
        print(f"  wrote {name}  ({len(out)} rows, {nfeat} features)")
    print(f"  skipped/unreadable: {missing}")


if __name__ == "__main__":
    main()