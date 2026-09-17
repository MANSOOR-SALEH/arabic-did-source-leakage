#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad_validation.py
=================
WebRTC validation of the non-speech control on AYDID.

The earlier version concatenated non-contiguous 10 ms slices into one waveform
before running the VAD. Every splice is a discontinuity -- audible as rasping,
and a broadband transient of exactly the kind a VAD calls speech. That artifact
exists only in the validator; the feature extractor selects spectral frames and
averages them, never concatenating anything. So the earlier 0.779 measured
damage introduced by the test rather than speech present in the data.

This version never concatenates. It runs the VAD over the ORIGINAL continuous
audio, producing a speech/non-speech decision per 30 ms window, then asks:

    of the frames the feature extractor RETAINS (bottom PCTL% by RMS),
    what fraction fall inside VAD-detected speech?

Apples to apples: same timeline, same frames, no splices.

Also reported:
  - the same overlap for the top-PCTL% (speech) frames, as a sanity check that
    the VAD is working at all -- this should be high
  - overlap under three VAD aggressiveness settings, since a conclusion that
    holds at all three is much harder to argue with
  - an ALTERNATIVE control: VAD-defined silence with a minimum run length, so
    brief inter-word gaps do not qualify. Reports how much audio survives, i.e.
    whether a clean control is constructible at all.
  - per-label breakdown: uneven contamination would mean the "non-speech"
    features could be reading how much speech survived per class.

SADA_S WebRTC columns of the paper's Table 6 are produced by v4_reruns.py R2,
not here.

Usage:
    python -m analysis.vad_validation --corpus aydid --n 300

Output: results/vad_validation_aydid.txt / .csv
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
from config import RESULTS

CORPORA = {
    "aydid": dict(
        manifest=str(AYDID_SOURCE_MANIFEST),
        wav_roots=[str(r) for r in AYDID_WAV_ROOTS],
        path_col="file_path", name_col="wav_name", label_col="dialect"),
}

SR = 16000
N_FFT, HOP = 512, 160          # 10 ms hop, matching the feature extractor
VAD_MS = 30                    # webrtcvad accepts 10/20/30 ms
PCTL = 30
MIN_SILENCE_MS = 200           # for the alternative control


def find_audio(row, cfg):
    fp = str(row.get(cfg["path_col"], "") or "")
    nm = str(row.get(cfg["name_col"], "") or "")
    for c in [fp] + [os.path.join(r, fp) for r in cfg["wav_roots"]] \
                  + [os.path.join(r, nm) for r in cfg["wav_roots"]]:
        if c and os.path.isfile(c):
            return c
    return None


def vad_timeline(y, aggressiveness):
    """Speech decision per VAD window over the CONTINUOUS original audio."""
    import webrtcvad
    v = webrtcvad.Vad(aggressiveness)
    pcm = np.clip(y * 32767, -32768, 32767).astype(np.int16).tobytes()
    win_samples = int(SR * VAD_MS / 1000)
    win_bytes = win_samples * 2
    n = len(pcm) // win_bytes
    dec = np.zeros(n, bool)
    for i in range(n):
        dec[i] = v.is_speech(pcm[i * win_bytes:(i + 1) * win_bytes], SR)
    return dec, win_samples


def frame_to_vad(n_frames, dec, win_samples):
    """Map each HOP-spaced analysis frame onto its VAD window."""
    centers = np.arange(n_frames) * HOP + HOP // 2
    idx = np.clip(centers // win_samples, 0, max(len(dec) - 1, 0))
    return dec[idx] if len(dec) else np.zeros(n_frames, bool)


def analyse(path, aggr_levels=(0, 2, 3)):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) < N_FFT * 3:
        return None
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
    n = len(rms)
    lo, hi = np.percentile(rms, PCTL), np.percentile(rms, 100 - PCTL)
    keep_low, keep_high = rms <= lo, rms >= hi

    out = {"dur": len(y) / SR, "n_frames": n,
           "frac_low": float(keep_low.mean())}
    for a in aggr_levels:
        dec, ws = vad_timeline(y, a)
        fv = frame_to_vad(n, dec, ws)
        out[f"low_speech_a{a}"] = float(fv[keep_low].mean()) if keep_low.any() else np.nan
        out[f"high_speech_a{a}"] = float(fv[keep_high].mean()) if keep_high.any() else np.nan
        out[f"overall_speech_a{a}"] = float(fv.mean())
        if a == 2:
            # alternative control: silence runs of at least MIN_SILENCE_MS
            sil = ~fv
            runs, start = [], None
            for i, s in enumerate(sil):
                if s and start is None:
                    start = i
                elif not s and start is not None:
                    runs.append((start, i)); start = None
            if start is not None:
                runs.append((start, len(sil)))
            min_frames = int(MIN_SILENCE_MS / (HOP / SR * 1000))
            good = [r for r in runs if r[1] - r[0] >= min_frames]
            out["alt_frames"] = int(sum(r[1] - r[0] for r in good))
            out["alt_frac"] = out["alt_frames"] / n
            out["alt_runs"] = len(good)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), default="aydid")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = CORPORA[args.corpus]
    man = pd.read_csv(cfg["manifest"]).sample(
        n=min(args.n, len(pd.read_csv(cfg["manifest"]))), random_state=args.seed)

    rows, missing = [], 0
    for i, r in enumerate(man.to_dict("records")):
        p = find_audio(r, cfg)
        if p is None:
            missing += 1; continue
        try:
            a = analyse(p)
        except Exception:
            missing += 1; continue
        if a is None:
            missing += 1; continue
        rows.append(dict(wav=r.get(cfg["name_col"]), label=r.get(cfg["label_col"]), **a))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(man)} ...")

    df = pd.DataFrame(rows)
    csv_path = RESULTS / f"vad_validation_{args.corpus}.csv"
    txt_path = RESULTS / f"vad_validation_{args.corpus}.txt"
    df.to_csv(csv_path, index=False)

    L = [f"NON-SPEECH CONTROL - FRAME-ALIGNED VALIDATION ({args.corpus.upper()})",
         "=" * 76,
         "  VAD runs on the ORIGINAL continuous audio; no concatenation, so no",
         "  splice artifacts. Question: of the frames the extractor retains",
         f"  (bottom {PCTL}% by RMS), what fraction fall inside VAD speech?",
         "",
         f"  utterances: {len(df)}   unreadable: {missing}",
         f"  mean duration: {df.dur.mean():.2f}s   "
         f"mean frames retained: {df.frac_low.mean():.3f}"]

    L.append("\n1  SPEECH OVERLAP BY VAD AGGRESSIVENESS")
    L.append("=" * 76)
    L.append(f"{'aggr':>5} {'low-energy':>12} {'high-energy':>13} {'whole utt':>11}")
    for a in (0, 2, 3):
        L.append(f"{a:>5} {df[f'low_speech_a{a}'].mean():>12.3f} "
                 f"{df[f'high_speech_a{a}'].mean():>13.3f} "
                 f"{df[f'overall_speech_a{a}'].mean():>11.3f}")
    L.append("")
    L.append("  high-energy should be HIGH - that is the sanity check that the VAD")
    L.append("  is working. low-energy is the control's contamination rate.")

    lo3 = df["low_speech_a3"].mean()
    lo0 = df["low_speech_a0"].mean()
    L.append("")
    if lo3 < 0.10 and lo0 < 0.25:
        L.append("  >>> CONTROL SUPPORTED: little speech in the retained frames at any")
        L.append("      setting. The earlier 0.779 was the concatenation artifact.")
    elif lo3 < 0.25:
        L.append("  >>> PARTIAL: strict VAD finds limited speech, permissive finds more.")
        L.append("      Report both, and consider the alternative control below.")
    else:
        L.append("  >>> CONTAMINATED: speech is present on the original timeline, so")
        L.append("      this is not an artifact of the test. Rebuild the control.")

    L.append("\n2  ALTERNATIVE CONTROL - VAD SILENCE RUNS >= "
             f"{MIN_SILENCE_MS} ms")
    L.append("=" * 76)
    if "alt_frac" in df:
        L.append(f"  mean fraction of each utterance qualifying : {df.alt_frac.mean():.3f}")
        L.append(f"  mean qualifying runs per utterance          : {df.alt_runs.mean():.1f}")
        L.append(f"  utterances with NO qualifying silence       : "
                 f"{(df.alt_runs == 0).mean()*100:.1f}%")
        if df.alt_frac.mean() < 0.05:
            L.append("  >>> too little true silence to build a control from. The corpus")
            L.append("      is tightly segmented; a non-speech control may not be")
            L.append("      constructible here, which is itself worth reporting.")
        else:
            L.append("  >>> enough true silence to rebuild the control on VAD-defined")
            L.append("      regions rather than an energy percentile.")

    L.append("\n3  BY LABEL")
    L.append("=" * 76)
    by = df.groupby("label").agg(
        low_a0=("low_speech_a0", "mean"), low_a3=("low_speech_a3", "mean"),
        alt=("alt_frac", "mean"), n=("dur", "size"))
    L.append(by.to_string())
    spread = by["low_a3"].max() - by["low_a3"].min()
    L.append(f"\n  spread across labels (strict VAD): {spread:.3f}")
    if spread > 0.10:
        L.append("  >>> uneven: the 'non-speech' features may partly be reading how")
        L.append("      much speech survived per class, not recording condition.")
    else:
        L.append("  >>> even across labels: contamination cannot by itself explain")
        L.append("      above-chance classification.")

    rep = "\n".join(L)
    print("\n" + rep)
    open(txt_path, "w", encoding="utf-8").write(rep + "\n")
    print(f"\n[done] wrote {csv_path}")


if __name__ == "__main__":
    main()