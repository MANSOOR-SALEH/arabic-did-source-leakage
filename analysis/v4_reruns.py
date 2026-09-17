#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4_reruns.py
============
Three reviewer-requested checks, all reported in the paper.

R1  DIMENSIONALITY-MATCHED ENCODER COMPARISON  (Section 5.3, Table 7 PCA column)
    All reviewers flag that the encoder arm has 2,048 features and the acoustic
    arm 61, so representation is confounded with capacity. We project the
    encoder features to 61 dimensions with PCA fitted on the TRAINING partition
    of each fold only, and rerun the comparison. Fitting PCA on the training
    split alone matters: fitting it on all data would leak test-set structure
    into the projection.

R2  SECOND DETECTOR FOR THE LOW-ENERGY CONTROL  (Section 5.2, Table 6 Silero
    columns and Figure 4). Two reviewers note the circularity of using an
    energy-sensitive VAD to adjudicate energy-selected frames. We add Silero
    VAD, a neural detector, and extend the analysis to SADA. Agreement between
    two detectors of different design is what the claim needs; disagreement
    would mean the finding is about WebRTC rather than about the frames.

R3  SELECTION-BIAS AUDIT OF THE RESTRICTED SADA SUBSET  (Section 3.2)
    Two reviewers question discarding 76% of SADA to obtain a speaker-disjoint
    baseline. We compare the restricted subset against the full one on every
    dimension that could bias the result: class balance, programme coverage,
    clip duration, and per-programme composition.

Usage:
    python -m analysis.v4_reruns                 # all three
    python -m analysis.v4_reruns --only R1
    python -m analysis.v4_reruns --only R2 --n 200

Output: results/v4_reruns_report.txt
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
from config import RESULTS

CHANNEL = {
    "aydid": str(FEATURES_CACHE / "aydid_channel_features_full.csv"),
    "sada":  str(FEATURES_CACHE / "sada_channel_features_full.csv"),
}
CHAN_META = ("wav_name", "clip_id", "speaker", "dialect", "region", "country",
             "source", "show_name", "environment", "domain", "duration", "stem")
REPORT = RESULTS / "v4_reruns_report.txt"
N_FOLDS = 20
SEED = 0


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]


def macro_f1(y, p, k):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0)


def fit_predict(X, y, tr, te, seed=0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    s = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=3000, random_state=seed).fit(
        s.transform(X[tr]), y[tr])
    return m.predict(s.transform(X[te]))


def fit_predict_pca(X, y, tr, te, ndim, seed=0):
    """PCA fitted on the training partition only, then the same classifier."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    s = StandardScaler().fit(X[tr])
    Xtr, Xte = s.transform(X[tr]), s.transform(X[te])
    p = PCA(n_components=min(ndim, Xtr.shape[1], len(Xtr) - 1),
            random_state=seed).fit(Xtr)
    m = LogisticRegression(max_iter=3000, random_state=seed).fit(
        p.transform(Xtr), y[tr])
    return m.predict(p.transform(Xte)), float(p.explained_variance_ratio_.sum())


# ---------------------------------------------------------------- R1
def r1(L, folds):
    from scipy import stats
    L.append("\n" + "=" * 78)
    L.append("R1  ENCODER vs ACOUSTIC, DIMENSIONALITY MATCHED (Section 5.3, Table 7)")
    L.append("=" * 78)
    L.append("  Encoder features projected to 61 dimensions by PCA fitted on the")
    L.append("  training partition of each fold only. Same folds, same classifier.")

    for corpus, chan in CHANNEL.items():
        if not os.path.exists(chan):
            L.append(f"\n  [{corpus}] {chan} missing - skipped")
            continue
        ch = pd.read_csv(chan)
        keycol = "wav_name" if "wav_name" in ch.columns else "clip_id"
        ch["stem"] = ch[keycol].apply(stem)
        fc = [c for c in ch.columns if c not in CHAN_META and c != "stem"
              and pd.api.types.is_numeric_dtype(ch[c])]

        for enc in ("wavlm", "xlsr"):
            npz = FEATURES_CACHE / f"enc_{corpus}_{enc}.npz"
            if not os.path.exists(npz):
                L.append(f"\n  [{corpus}/{enc}] {npz} missing - skipped")
                continue
            z = np.load(npz, allow_pickle=True)
            d = pd.DataFrame({"stem": [stem(k) for k in z["keys"]],
                              "label": [str(v) for v in z["label"]],
                              "speaker": [str(v) for v in z["speaker"]],
                              "source": [str(v) for v in z["source"]]})
            d["_row"] = np.arange(len(d))
            m = d.merge(ch[["stem"] + fc].drop_duplicates("stem"), on="stem",
                        how="inner").reset_index(drop=True)
            if len(m) == 0:
                L.append(f"\n  [{corpus}/{enc}] merge empty - skipped")
                continue
            E = z["features"][m["_row"].values]
            C = np.nan_to_num(m[fc].values.astype(float))
            labs = sorted(m.label.unique()); l2i = {l: i for i, l in enumerate(labs)}
            y = m.label.map(l2i).values
            src = m.source.astype(str).values
            k = len(labs); n = len(m)
            src_by_lab = {l: sorted(np.unique(src[y == l2i[l]])) for l in labs}
            if any(len(v) < 2 for v in src_by_lab.values()):
                L.append(f"\n  [{corpus}/{enc}] a label has one source - skipped")
                continue

            e_raw, e_pca, c_raw, evr = [], [], [], []
            for f in range(folds):
                r = np.random.RandomState(f)
                held = [pool[r.randint(len(pool))] for pool in src_by_lab.values()]
                te = np.isin(src, held); tr = ~te
                if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k:
                    continue
                e_raw.append(macro_f1(y[te], fit_predict(E, y, tr, te, f), k))
                pp, v = fit_predict_pca(E, y, tr, te, len(fc), f)
                e_pca.append(macro_f1(y[te], pp, k)); evr.append(v)
                c_raw.append(macro_f1(y[te], fit_predict(C, y, tr, te, f), k))
            if len(e_pca) < 3:
                L.append(f"\n  [{corpus}/{enc}] too few usable folds")
                continue
            a, b, c = np.array(e_raw), np.array(e_pca), np.array(c_raw)
            nn = min(len(a), len(b), len(c))
            d_raw = a[:nn] - c[:nn]
            d_pca = b[:nn] - c[:nn]

            def ci(v):
                se = v.std(ddof=1) / np.sqrt(len(v))
                t = stats.t.ppf(0.975, len(v) - 1)
                return v.mean() - t * se, v.mean() + t * se

            lo1, hi1 = ci(d_raw); lo2, hi2 = ci(d_pca)
            L.append(f"\n  [{corpus} / {enc}]  n={n}  acoustic dims={len(fc)}  "
                     f"folds={nn}")
            L.append(f"    acoustic            : {c[:nn].mean():.3f}")
            L.append(f"    encoder, 2048 dims  : {a[:nn].mean():.3f}   "
                     f"diff {d_raw.mean():+.3f}  CI [{lo1:+.3f}, {hi1:+.3f}]  "
                     f"{(d_raw>0).sum()}/{nn}")
            L.append(f"    encoder, {len(fc)} dims (PCA): {b[:nn].mean():.3f}   "
                     f"diff {d_pca.mean():+.3f}  CI [{lo2:+.3f}, {hi2:+.3f}]  "
                     f"{(d_pca>0).sum()}/{nn}")
            L.append(f"    variance retained by PCA : {np.mean(evr):.3f}")
            shrink = d_raw.mean() - d_pca.mean()
            L.append(f"    advantage attributable to dimensionality : {shrink:+.3f}")
            if lo2 > 0:
                L.append("    >>> the encoder advantage SURVIVES dimensionality matching")
            elif hi2 < 0:
                L.append("    >>> the acoustic features BEAT the matched encoder")
            else:
                L.append("    >>> no detectable advantage once dimensionality is matched;")
                L.append("        the Section 5.3 result was a capacity effect")


# ---------------------------------------------------------------- R2
def r2(L, n_utts):
    L.append("\n" + "=" * 78)
    L.append("R2  SECOND DETECTOR FOR THE LOW-ENERGY CONTROL (Section 5.2, Table 6)")
    L.append("=" * 78)
    try:
        import torch
        import librosa
        import webrtcvad
    except Exception as e:
        L.append(f"  missing dependency ({type(e).__name__}) - skipped")
        return
    try:
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            model = load_silero_vad()
            get_ts = get_speech_timestamps
        except ImportError:
            model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                                              trust_repo=True, onnx=False)
            get_ts = utils[0]
    except Exception as e:
        L.append(f"  could not load Silero VAD ({type(e).__name__}: {e})")
        L.append("  Install with internet access, or report WebRTC alone and state")
        L.append("  the single-detector limitation explicitly.")
        return
    SR, N_FFT, HOP, PCTL = 16000, 512, 160, 30
    CORPORA = {
        "AYDID": (str(AYDID_SOURCE_MANIFEST),
                  [str(r) for r in AYDID_WAV_ROOTS], "file_path", "dialect"),
        "SADA":  (str(SADA_SOURCE_MANIFEST),
                  [str(r) for r in SADA_WAV_ROOTS], "file_path", "dialect"),
    }
    for name, (man_p, roots, pcol, lcol) in CORPORA.items():
        if not os.path.exists(man_p):
            L.append(f"\n  [{name}] manifest not found - skipped")
            continue
        man = pd.read_csv(man_p, low_memory=False)
        man.columns = [c.strip() for c in man.columns]
        man = man.sample(min(n_utts, len(man)), random_state=SEED)
        rows, miss = [], 0
        for r in man.to_dict("records"):
            p = None
            for cand in [str(r.get(pcol, ""))] + \
                        [os.path.join(rt, str(r.get(pcol, ""))) for rt in roots]:
                if cand and os.path.isfile(cand):
                    p = cand; break
            if p is None:
                miss += 1; continue
            try:
                y, _ = librosa.load(p, sr=SR, mono=True)
                if len(y) < N_FFT * 3:
                    miss += 1; continue
                S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
                rms = librosa.feature.rms(S=S, frame_length=N_FFT, hop_length=HOP)[0]
                keep = rms <= np.percentile(rms, PCTL)
                nf = len(rms)
                centres = np.arange(nf) * HOP + HOP // 2

                # Silero speech timeline
                ts = get_ts(torch.from_numpy(y), model, sampling_rate=SR)
                sil = np.zeros(len(y), bool)
                for seg in ts:
                    sil[seg['start']:seg['end']] = True
                sv = sil[np.clip(centres, 0, len(y) - 1)]

                # WebRTC timeline
                v = webrtcvad.Vad(3)
                pcm = np.clip(y * 32767, -32768, 32767).astype(np.int16).tobytes()
                w = int(SR * 0.03); wb = w * 2
                dec = np.array([v.is_speech(pcm[i*wb:(i+1)*wb], SR)
                                for i in range(len(pcm)//wb)], bool)
                wv = dec[np.clip(centres // w, 0, max(len(dec)-1, 0))] \
                     if len(dec) else np.zeros(nf, bool)

                rows.append(dict(label=r.get(lcol),
                                 silero_low=float(sv[keep].mean()),
                                 silero_high=float(sv[~keep].mean()),
                                 webrtc_low=float(wv[keep].mean())))
            except Exception:
                miss += 1
        if not rows:
            L.append(f"\n  [{name}] no usable audio - skipped")
            continue
        df = pd.DataFrame(rows)
        L.append(f"\n  [{name}]  n={len(df)}  unreadable={miss}")
        L.append(f"    Silero  low-energy frames in speech : {df.silero_low.mean():.3f}")
        L.append(f"    WebRTC(3) low-energy frames in speech: {df.webrtc_low.mean():.3f}")
        L.append(f"    Silero  high-energy (positive control): {df.silero_high.mean():.3f}")
        by = df.groupby("label")["silero_low"].mean()
        L.append(f"    by label (Silero): "
                 f"{ {str(k): round(v,3) for k, v in by.items()} }")
        L.append(f"    spread across labels: {by.max()-by.min():.3f}")
        agree = abs(df.silero_low.mean() - df.webrtc_low.mean())
        if agree < 0.15:
            L.append("    >>> the two detectors agree; the contamination finding is not")
            L.append("        an artefact of WebRTC's energy sensitivity.")
        else:
            L.append("    >>> the detectors disagree substantially. Report both and")
            L.append("        weaken the claim to a WebRTC-specific observation.")


# ---------------------------------------------------------------- R3
def r3(L, min_clips):
    L.append("\n" + "=" * 78)
    L.append("R3  SELECTION-BIAS AUDIT OF THE RESTRICTED SADA SUBSET (Section 3.2)")
    L.append("=" * 78)
    path = CHANNEL["sada"]
    if not os.path.exists(path):
        L.append(f"  {path} missing - skipped")
        return
    d = pd.read_csv(path).dropna(subset=["dialect", "show_name", "speaker"])
    cnt = d.groupby("speaker").size()
    keep = set(cnt[cnt >= min_clips].index)
    r = d[d.speaker.isin(keep)]
    L.append(f"  full: {len(d)} clips, {d.speaker.nunique()} units, "
             f"{d.show_name.nunique()} programmes")
    L.append(f"  restricted (>= {min_clips} clips/unit): {len(r)} clips "
             f"({100*len(r)/len(d):.0f}%), {r.speaker.nunique()} units, "
             f"{r.show_name.nunique()} programmes")

    L.append("\n  class balance:")
    fb = d.dialect.value_counts(normalize=True)
    rb = r.dialect.value_counts(normalize=True)
    for l in sorted(d.dialect.unique()):
        L.append(f"    {l:<12} full {fb.get(l,0):.3f}   restricted {rb.get(l,0):.3f}   "
                 f"delta {rb.get(l,0)-fb.get(l,0):+.3f}")
    maxd = max(abs(rb.get(l,0)-fb.get(l,0)) for l in d.dialect.unique())
    L.append(f"    largest shift: {maxd:.3f}")

    L.append("\n  programme coverage:")
    fp = d.show_name.value_counts(normalize=True)
    rp = r.show_name.value_counts(normalize=True)
    L.append(f"    programmes retained: {len(rp)}/{len(fp)}")
    L.append(f"    share of clips from the largest programme: "
             f"full {fp.max():.3f}, restricted {rp.max():.3f}")
    if "duration" in d.columns:
        L.append(f"\n  duration: full mean {d.duration.astype(float).mean():.2f}s, "
                 f"restricted {r.duration.astype(float).mean():.2f}s")

    L.append("")
    if maxd < 0.05 and len(rp) >= 0.5 * len(fp):
        L.append("  >>> the restricted subset preserves class balance and most")
        L.append("      programmes. Report this table; the selection is defensible.")
    else:
        L.append("  >>> the restricted subset differs materially from the full one.")
        L.append("      Report the shift explicitly and qualify the increment as")
        L.append("      applying to that subset rather than to SADA as a whole.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["R1", "R2", "R3"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-clips", type=int, default=4)
    args = ap.parse_args()

    L = ["V4 RERUNS - THREE REVIEWER-REQUESTED CHECKS", "=" * 78]
    if args.only in (None, "R1"):
        r1(L, args.folds)
    if args.only in (None, "R3"):
        r3(L, args.min_clips)
    if args.only in (None, "R2"):
        r2(L, args.n)

    rep = "\n".join(L)
    print("\n" + rep)
    open(REPORT, "w", encoding="utf-8").write(rep + "\n")
    print(f"\n[done] wrote {REPORT}")


if __name__ == "__main__":
    main()