#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adi17_protocols.py
==================
ADI-17 as a third point on the random-to-source measure.

The manuscript reports that random-to-source drop for AYDID and SADA_S, and
every reviewer has made the same objection: two corpora are not a distribution,
and one of them is ours. ADI-17 is a published seventeen-dialect benchmark we
did not build, with 1,094 distinct source videos across 12,150 utterances and
between seventeen and one hundred and twenty-five videos per dialect, so the
protocol is constructible on the full published class set.

Two protocols, matched in test-set size:

    random          utterances shuffled; segments from one video may fall on
                    both sides, which is how the benchmark is normally used
    source-disjoint one whole video held out per dialect, so every class
                    survives on both sides and no held-out video contributes
                    to training

ADI-17 supplies no speaker labels, so there is no speaker-disjoint arm. This is
not a weakness for the present purpose: the random split is the maximum-leakage
case, and the random-to-source drop is precisely the quantity the manuscript
defines identically across corpora.

A note on an earlier attempt. A previous run of this analysis produced a
source-disjoint macro-F1 below chance, which we traced to holding out a pooled
random fraction of videos: with few videos per dialect, some dialects had no
test video in a fold and contributed zero to the macro average, driving the
score down for a bookkeeping reason rather than a leakage one. Holding out
exactly one video per dialect removes that failure, and the feasibility check
confirms every dialect has enough videos for it.

The low-energy variant is run alongside, giving a third corpus for the
contamination result as well.

Usage:
    python adi17_protocols.py
    python adi17_protocols.py --folds 30

Output: adi17_protocols_report.txt, adi17_protocols.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

FULL = "adi17_channel_features_full.csv"
NONSP = "adi17_channel_features_nonspeech.csv"
META = ("utt_id", "video", "dialect")
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


def run(path, tag, folds, L):
    if not os.path.exists(path):
        L.append(f"  [{tag}] {path} missing \u2014 skipped")
        return None
    d = pd.read_csv(path).dropna(subset=["dialect", "video"])
    fc = [c for c in d.columns if c not in META
          and pd.api.types.is_numeric_dtype(d[c])]
    X = np.nan_to_num(d[fc].values.astype(float))
    labs = sorted(d.dialect.unique()); l2i = {l: i for i, l in enumerate(labs)}
    y = d.dialect.map(l2i).values
    vid = d.video.astype(str).values
    k = len(labs); n = len(d); idx = np.arange(n)
    vids_by_lab = {l: sorted(np.unique(vid[y == l2i[l]])) for l in labs}

    L.append(f"\n  [{tag}] n={n}  dialects={k}  videos={d.video.nunique()}  "
             f"features={len(fc)}  chance={1/k:.3f}")
    thin = {l: len(v) for l, v in vids_by_lab.items() if len(v) < 2}
    if thin:
        L.append(f"  [{tag}] dialects with a single video: {thin} \u2014 excluded "
                 f"from the source arm")

    res = {}
    for proto in ("random", "source"):
        pf, sizes = [], []
        for f in range(folds):
            r = np.random.RandomState(f)
            if proto == "source":
                held = [pool[r.randint(len(pool))]
                        for pool in vids_by_lab.values() if len(pool) > 1]
                te = np.isin(vid, held)
            else:
                # match the source arm's typical test size
                nte = max(k * 2, int(0.06 * n))
                te = np.zeros(n, bool)
                te[r.choice(idx, size=nte, replace=False)] = True
            tr = ~te
            if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k or te.sum() == 0:
                continue
            pf.append(macro_f1(y[te], fit_predict(X, y, tr, te, f), k))
            sizes.append(int(te.sum()))
        if len(pf) < 3:
            L.append(f"  [{tag}] {proto}: too few usable folds")
            return None
        res[proto] = np.array(pf)
        L.append(f"  [{tag}] {proto:<7} {np.mean(pf):.3f} +/- {np.std(pf):.3f}   "
                 f"test n median {int(np.median(sizes))}   ({len(pf)} folds)")

    a, b = res["random"], res["source"]
    m = min(len(a), len(b))
    dd = a[:m] - b[:m]
    L.append(f"  [{tag}] random \u2192 source = {dd.mean():+.3f}  "
             f"({(dd > 0).sum()}/{m} folds, sd {dd.std():.3f})")
    if b.mean() < 1.0 / k:
        L.append(f"  [{tag}] !! source-disjoint score is below chance; check that "
                 f"every dialect has a test video in every fold")
    return dict(tag=tag, n=n, k=k, random=a.mean(), source=b.mean(),
                drop=dd.mean(), wins=int((dd > 0).sum()), folds=m,
                sd=dd.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    L = ["ADI-17 \u2014 RANDOM VERSUS SOURCE-DISJOINT", "=" * 74,
         "  Published seventeen-dialect benchmark, not built by the authors.",
         "  Source variable is the YouTube video id; one video held out per",
         "  dialect. No speaker labels exist, so the leaky arm is a random",
         "  split, which is the maximum-leakage case."]

    rows = []
    for path, tag in ((FULL, "full"), (NONSP, "low-energy")):
        r = run(path, tag, args.folds, L)
        if r:
            rows.append(r)

    L.append("\n" + "=" * 74)
    L.append("COMPARISON WITH THE OTHER TWO CORPORA")
    L.append("=" * 74)
    L.append("  random \u2192 source, acoustic features, full subsets:")
    L.append("    AYDID   7 classes   +0.065")
    L.append("    SADA_S  3 classes   +0.289")
    if rows:
        r = rows[0]
        L.append(f"    ADI-17 {r['k']:>2} classes   {r['drop']:+.3f}")
        L.append("")
        L.append("  Chance levels differ (0.143, 0.333, 0.059), so we also give the")
        L.append("  drop as a fraction of the accuracy available above chance:")
        for name, drop, rand, ch in (("AYDID", 0.065, 0.641, 1/7),
                                     ("SADA_S", 0.289, 0.523, 1/3),
                                     ("ADI-17", r["drop"], r["random"], 1/r["k"])):
            head = rand - ch
            L.append(f"    {name:<7} {drop/head:>6.2f} of headroom "
                     f"(drop {drop:+.3f}, random {rand:.3f}, chance {ch:.3f})")
        L.append("")
        if r["drop"] > 0.15:
            L.append("  >>> ADI-17 shows substantial source leakage, on a benchmark the")
            L.append("      authors did not build and at the published class set. The")
            L.append("      manuscript can report three corpora rather than two, which")
            L.append("      is the objection every reviewer has raised.")
        elif r["drop"] > 0.05:
            L.append("  >>> ADI-17 shows moderate source leakage, between the two")
            L.append("      Arabic corpora already reported. Report it as a third point")
            L.append("      and note that the range is continuous rather than binary.")
        else:
            L.append("  >>> ADI-17 shows little source leakage despite its structure.")
            L.append("      That is a substantive finding and should be reported as")
            L.append("      such: it weakens any suggestion that nested corpora leak by")
            L.append("      default.")

    if rows:
        pd.DataFrame(rows).to_csv("adi17_protocols.csv", index=False)
    rep = "\n".join(L)
    print("\n" + rep)
    open("adi17_protocols_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done]")


if __name__ == "__main__":
    main()
