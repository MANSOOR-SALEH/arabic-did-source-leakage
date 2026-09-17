#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adi17_features.py
=================
Extracts the same sixty-one non-temporal acoustic features used on AYDID and
SADA from the ADI-17 test split, so that ADI-17 can serve as a third point on
the random-to-source measure.

Why this corpus matters for the paper. AYDID and SADA are two Arabic corpora,
one of them ours, and every reviewer has raised the same objection: a contrast
between two corpora is not a distribution, and matching removes named confounds
but not corpus identity. ADI-17 is a published seventeen-dialect benchmark that
we did not build, and the feasibility check found every one of its seventeen
dialects carries between seventeen and one hundred and twenty-five distinct
source videos. A source-disjoint protocol is therefore constructible on the
full published class set.

Note on provenance. The record id has the form 1KidT1HxOyI_001677-002032; the
leading eleven characters are the YouTube video id, which is the source
variable. ADI-17 provides no speaker labels, so the leaky arm here is a random
utterance split rather than a speaker-disjoint one. That is the harder
comparison, not the easier one: a random split is the maximum-leakage case, and
the random-to-source drop is the quantity the manuscript already defines
identically across corpora.

Audio is read from the parquet bytes column shard by shard, so nothing is
written to disk beyond the feature table and no separate download step is
needed. Progress is saved after each shard and the script resumes, since the
network on this machine has dropped mid-fetch before.

Feature definition is identical to the AYDID and SADA extractors: 16 kHz mono,
512-sample window, 160-sample hop, 26 mel bands, statistics averaged over the
utterance so that no temporal phonetic sequence survives.

Usage:
    python adi17_features.py --shards 1        # smoke test, one shard
    python adi17_features.py                   # all five

Output: adi17_channel_features_full.csv, adi17_channel_features_nonspeech.csv
"""
import argparse
import io
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = ("https://huggingface.co/datasets/ArabicSpeech/ADI17/resolve/main/"
        "data/{split}-{i:05d}-of-{n:05d}.parquet")
SR = 16000
N_FFT, HOP = 512, 160
N_MELS, N_MFCC = 26, 20
PCTL = 30
OUT_FULL = "adi17_channel_features_full.csv"
OUT_NONSP = "adi17_channel_features_nonspeech.csv"


def frame_stats(S, mfcc, zcr, rms, keep):
    import librosa
    if keep.sum() < 3:
        keep = np.ones(S.shape[1], bool)
    Sk = S[:, keep]
    f = {}
    mel = librosa.feature.melspectrogram(S=Sk ** 2, sr=SR, n_mels=N_MELS)
    ltas = np.log(mel.mean(axis=1) + 1e-10)
    f.update({f"ltas{i:02d}": v for i, v in enumerate(ltas - ltas.mean())})
    for name, fn in [("centroid", librosa.feature.spectral_centroid),
                     ("rolloff", librosa.feature.spectral_rolloff),
                     ("bandwidth", librosa.feature.spectral_bandwidth),
                     ("flatness", librosa.feature.spectral_flatness)]:
        v = fn(S=Sk) if name == "flatness" else fn(S=Sk, sr=SR)
        f[f"{name}_mean"] = float(np.mean(v))
        f[f"{name}_sd"] = float(np.std(v))
    freqs = np.maximum(librosa.fft_frequencies(sr=SR, n_fft=N_FFT), 1.0)
    logp = np.log(Sk.mean(axis=1) + 1e-10)
    f["tilt"] = float(np.polyfit(np.log(freqs), logp, 1)[0])
    e = rms[keep] if keep.sum() >= 3 else rms
    nf = float(np.percentile(e, 10) + 1e-10)
    f["noise_floor"] = float(20 * np.log10(nf))
    f["snr"] = float(20 * np.log10((np.percentile(e, 90) + 1e-10) / nf))
    f["zcr_mean"] = float(np.mean(zcr[keep]))
    f["zcr_sd"] = float(np.std(zcr[keep]))
    f["rms_mean"] = float(np.mean(e))
    f["rms_sd"] = float(np.std(e))
    mk = mfcc[:, keep]
    f.update({f"mfcc{i:02d}": float(v) for i, v in enumerate(mk.mean(axis=1))})
    return f


def extract(wav_bytes):
    import librosa
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < N_FFT * 2:
        return None, None
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT,
                                             hop_length=HOP)[0]
    n = min(S.shape[1], len(rms), len(zcr))
    S, rms, zcr = S[:, :n], rms[:n], zcr[:n]
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS)),
        n_mfcc=N_MFCC)
    lo = np.percentile(rms, PCTL)
    return (frame_stats(S, mfcc, zcr, rms, np.ones(n, bool)),
            frame_stats(S, mfcc, zcr, rms, rms <= lo))


def read_shard(url, retries=5):
    for a in range(retries):
        try:
            return pd.read_parquet(url)
        except Exception as e:
            if a == retries - 1:
                raise
            w = 2 ** a
            print(f"    retry {a+1} after {type(e).__name__}; waiting {w}s",
                  flush=True)
            time.sleep(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--total", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many utterances per shard")
    args = ap.parse_args()

    done = set()
    if os.path.exists(OUT_FULL):
        prev = pd.read_csv(OUT_FULL)
        done = set(prev["utt_id"].astype(str))
        print(f"resuming: {len(done)} utterances already extracted")

    rows_full, rows_nonsp, skipped = [], [], 0
    for i in range(args.shards):
        url = BASE.format(split=args.split, i=i, n=args.total)
        print(f"shard {i} ...", flush=True)
        try:
            d = read_shard(url)
        except Exception as e:
            print(f"  shard {i} failed: {type(e).__name__}: {e}")
            continue
        if args.limit:
            d = d.head(args.limit)
        for j, r in enumerate(d.itertuples(index=False)):
            uid = str(r.id)
            if uid in done:
                continue
            try:
                a, b = extract(r.audio["bytes"])
            except Exception:
                skipped += 1
                continue
            if a is None:
                skipped += 1
                continue
            meta = {"utt_id": uid, "video": uid[:11], "dialect": str(r.dialect)}
            rows_full.append({**meta, **a})
            rows_nonsp.append({**meta, **b})
            if (j + 1) % 500 == 0:
                print(f"  {j+1}/{len(d)} ...", flush=True)
        # save after every shard so a dropped connection costs one shard
        for path, rows in ((OUT_FULL, rows_full), (OUT_NONSP, rows_nonsp)):
            if rows:
                df = pd.DataFrame(rows)
                if os.path.exists(path):
                    df = pd.concat([pd.read_csv(path), df], ignore_index=True)
                    df = df.drop_duplicates("utt_id")
                df.to_csv(path, index=False)
        rows_full, rows_nonsp = [], []
        print(f"  shard {i} saved", flush=True)

    if os.path.exists(OUT_FULL):
        f = pd.read_csv(OUT_FULL)
        META = ("utt_id", "video", "dialect")
        print(f"\n{OUT_FULL}: {len(f)} utterances, "
              f"{len([c for c in f.columns if c not in META])} features")
        print(f"  dialects {f.dialect.nunique()}   videos {f.video.nunique()}")
        g = f.groupby("dialect").agg(n=("utt_id", "size"),
                                     videos=("video", "nunique"))
        print(g.to_string())
    print(f"  unreadable/skipped: {skipped}")


if __name__ == "__main__":
    main()
