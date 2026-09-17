#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocols_sada.py
=================
The counterfactual corpus.

SADA_S has 63 shows and 82.5% of them carry more than one dialect. So holding
out whole shows removes recording provenance while EVERY dialect survives in
training from other shows. There is no country confound and no nesting -- the
direct answer to the objection that source and country are the same variable.

Three protocols, size-matched:

    random          utterances shuffled
    speaker         speakers held out WITHIN shows (shows shared)
    show            whole shows held out

Speaker unit is the FileName:Speaker composite. Note SADA_S averages ~1.7 clips
per speaker, so the speaker-disjoint arm is close to a random split there; the
report states the clips-per-speaker figure so this is visible.

Usage:
    python -m analysis.protocols_sada
    python -m analysis.protocols_sada --variants full nonspeech --folds 20

Output: results/protocols_sada_report.txt, results/protocols_sada_perfold.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FEATURES_CACHE
from config import RESULTS

FEAT = str(FEATURES_CACHE / "sada_channel_features_{}.csv")
PERFOLD = RESULTS / "protocols_sada_perfold.csv"
REPORT = RESULTS / "protocols_sada_report.txt"

META = ("wav_name", "speaker", "dialect", "show_name", "environment", "duration")
N_FOLDS = 20
SEED = 0


def macro_f1(y, p, k):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0)


def fit_predict(X, y, tr, te, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    s = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=3000, random_state=seed).fit(
        s.transform(X[tr]), y[tr])
    return m.predict(s.transform(X[te]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["full", "nonspeech", "cmvn"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    from scipy import stats

    L = ["SADA_S - THREE PROTOCOLS (CROSSED-SOURCE CORPUS)", "=" * 78,
         "  63 shows, most carrying several dialects. Holding out shows removes",
         "  provenance while every dialect survives in training from other shows.",
         "  No country confound, no nesting.", ""]

    perfold = []
    for variant in args.variants:
        path = FEAT.format(variant)
        if not os.path.exists(path):
            L.append(f"  {path} missing - run features.extract_acoustic_sada first")
            continue
        d = pd.read_csv(path).dropna(subset=["dialect", "show_name", "speaker"])
        fcols = [c for c in d.columns
                 if c not in META and pd.api.types.is_numeric_dtype(d[c])]
        X = np.nan_to_num(d[fcols].values.astype(float))
        labs = sorted(d.dialect.unique())
        l2i = {l: i for i, l in enumerate(labs)}
        y = d.dialect.map(l2i).values
        shw = d.show_name.astype(str).values
        spk = d.speaker.astype(str).values
        idx = np.arange(len(d))
        k = len(labs)

        if variant == args.variants[0]:
            ct = pd.crosstab(d.show_name, d.dialect)
            crossed = (ct > 0).sum(axis=1)
            cps = d.groupby("speaker").size()
            L.append(f"  n={len(d)}  dialects={k}  shows={len(ct)}  "
                     f"speaker units={d.speaker.nunique()}")
            L.append(f"  chance macro-F1 = {1/k:.3f}")
            L.append(f"  shows spanning >1 dialect: {int((crossed>1).sum())}/{len(ct)} "
                     f"({100*(crossed>1).mean():.1f}%)")
            L.append(f"  clips per speaker unit: median={cps.median():.1f} "
                     f"mean={cps.mean():.1f}")
            if cps.median() < 3:
                L.append("  >>> few clips per speaker: the speaker-disjoint arm is close")
                L.append("      to a random split here. Read that arm with care.")
            L.append("")

        shows = sorted(np.unique(shw))
        spk_by_show = {s: sorted(np.unique(spk[shw == s])) for s in shows}

        # target test size from the show arm (hold out ~25% of shows)
        n_hold = max(1, int(round(0.25 * len(shows))))

        L.append(f"{'=' * 78}\n[{variant}]  features={len(fcols)}\n{'=' * 78}")
        res = {}
        for proto in ["random", "speaker", "show"]:
            pf, ty, tp = [], [], []
            for f in range(args.folds):
                r = np.random.RandomState(f)
                if proto == "random":
                    te = np.zeros(len(d), bool)
                    te[r.choice(idx, size=len(d) // 4, replace=False)] = True
                elif proto == "speaker":
                    held = []
                    for s in shows:
                        sp = spk_by_show[s]
                        if len(sp) < 2:
                            continue
                        kk = min(max(1, int(round(0.25 * len(sp)))), len(sp) - 1)
                        held.extend(r.choice(sp, size=kk, replace=False).tolist())
                    te = np.isin(spk, held)
                else:
                    hs = r.choice(shows, size=n_hold, replace=False)
                    te = np.isin(shw, hs)
                tr = ~te
                if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k \
                        or te.sum() == 0:
                    continue
                p = fit_predict(X, y, tr, te, f)
                pf.append(macro_f1(y[te], p, k))
                ty.append(y[te]); tp.append(p)
                perfold.append(dict(variant=variant, protocol=proto, fold=f,
                                    n_train=int(tr.sum()), n_test=int(te.sum()),
                                    macro_f1=pf[-1]))
            if not pf:
                L.append(f"  {proto:<9} no usable folds")
                continue
            pooled = macro_f1(np.concatenate(ty), np.concatenate(tp), k)
            res[proto] = np.array(pf)
            L.append(f"  {proto:<9} mean-of-fold={np.mean(pf):.3f} "
                     f"+/-{np.std(pf):.3f}   POOLED={pooled:.3f}   (n={len(pf)})")

        if all(p in res for p in ("random", "speaker", "show")):
            r_, s_, h_ = res["random"], res["speaker"], res["show"]
            n = min(len(r_), len(s_), len(h_))
            d1 = r_[:n] - s_[:n]
            d2 = s_[:n] - h_[:n]
            t, p = stats.ttest_rel(s_[:n], h_[:n])
            ci = stats.t.interval(0.95, n - 1, loc=d2.mean(),
                                  scale=d2.std(ddof=1) / np.sqrt(n))
            L.append("")
            L.append(f"  random -> speaker : {d1.mean():+.3f}")
            L.append(f"  speaker -> show   : {d2.mean():+.3f}  "
                     f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.2e}  "
                     f"wins {(d2>0).sum()}/{n}")
            if variant == "full":
                L.append("")
                L.append(f"  SADA_S show holdout : {d2.mean():+.3f}  (CROSSED)")
                L.append("")
                if ci[1] < 0.05:
                    L.append("  >>> As predicted: a crossed corpus does not leak. Holding")
                    L.append("      out shows costs nothing because no show signature")
                    L.append("      predicts the label. This is the counterfactual that")
                    L.append("      shows the effect is corpus construction, not the task.")
                elif d2.mean() > 0.10:
                    L.append("  >>> Leaks despite being crossed. Then crossing is NOT")
                    L.append("      sufficient protection, and the mechanism is acoustic")
                    L.append("      source distinctiveness rather than label-source")
                    L.append("      correlation. Report this - it is the more surprising")
                    L.append("      result and it reframes the prescription.")
                else:
                    L.append("  >>> Small but non-zero. Report the CI and treat SADA_S as")
                    L.append("      the low-leakage anchor of the range.")

    pd.DataFrame(perfold).to_csv(PERFOLD, index=False)
    rep = "\n".join(L)
    print("\n" + rep)
    open(REPORT, "w", encoding="utf-8").write(rep + "\n")
    print(f"\n[done] wrote {PERFOLD}")


if __name__ == "__main__":
    main()