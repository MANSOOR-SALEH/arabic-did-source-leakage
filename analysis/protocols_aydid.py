#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocols_aydid.py
==================
The AYDID result that answers TACL objection (a).

All 40 shows are Yemeni, so country does not vary. Any drop from
speaker-disjoint to show-disjoint therefore cannot be "you held out a whole
country" -- the confound SADIC cannot escape.

Three protocols, all with IDENTICAL train and test sizes:

    random           utterances shuffled; speaker and show shared
    speaker-disjoint speakers held out WITHIN shows; all 40 shows both sides
    show-disjoint    one whole show per dialect held out

Because the arms are size-matched by subsampling, no part of any gap is
attributable to training-set size. The speaker arm keeps every show in
training, so the ONLY difference between the speaker and show arms is whether
show identity is shared.

Four feature variants are run (full / speech / nonspeech / cmvn), so the
non-phonetic control is measured under the same protocols as the main result.

Reports BOTH conventions -- mean-of-fold and pooled predictions -- because
mixing them is what produced the Table 4 error. Never compare across them.

Usage:
    python -m analysis.protocols_aydid
    python -m analysis.protocols_aydid --variants full nonspeech --folds 10

Output: results/protocols_aydid_report.txt, results/protocols_aydid_perfold.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AYDID_SOURCE_MANIFEST as MANIFEST
from config import FEATURES_CACHE
from config import FOLDS
from config import RESULTS

DUR_CACHE = FEATURES_CACHE / "aydid_durations.csv"
FEAT = str(FEATURES_CACHE / "aydid_channel_features_{}.csv")
SHOW_FOLDS = FOLDS / "table2_source.csv"
PERFOLD = RESULTS / "protocols_aydid_perfold.csv"
REPORT = RESULTS / "protocols_aydid_report.txt"

MAX_DUR = 30.0
N_FOLDS = 20
SEED = 0
META = ("wav_name", "speaker", "dialect", "show_name", "duration")


def macro_f1(y, p, labels):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def fit_predict(X, y, tr, te, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    s = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=3000, random_state=seed)
    m.fit(s.transform(X[tr]), y[tr])
    return m.predict(s.transform(X[te]))


def match_sizes(tr_idx, te_idx, n_tr, n_te, rng):
    """Subsample so every protocol trains and tests on the same amount."""
    tr = rng.choice(tr_idx, size=min(n_tr, len(tr_idx)), replace=False)
    te = rng.choice(te_idx, size=min(n_te, len(te_idx)), replace=False)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["full", "speech", "nonspeech", "cmvn"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    if os.path.exists(DUR_CACHE):
        man = man.merge(pd.read_csv(DUR_CACHE), on="wav_name", how="left")
        man = man[man["duration"].isna() | (man["duration"] <= MAX_DUR)]

    # target sizes from the show-disjoint arm
    sf = pd.read_csv(SHOW_FOLDS)
    N_TE = int(sf["n_test"].min())
    N_TR = int(len(man) - sf["n_test"].max())

    L = ["AYDID - THREE PROTOCOLS, CHANNEL-ONLY FEATURES", "=" * 76,
         f"  clips={len(man)}  shows={man.show_name.nunique()}  "
         f"speakers={man.speaker_id.nunique()}  dialects={man.dialect.nunique()}",
         f"  chance macro-F1 = {1/man.dialect.nunique():.3f}",
         f"  size-matched across arms: train={N_TR}  test={N_TE}",
         f"  folds={args.folds}"]

    shows_by_lab = {lab: sorted(g.show_name.unique())
                    for lab, g in man.groupby("dialect")}
    spk_by_show = {s: sorted(g.speaker_id.unique())
                   for s, g in man.groupby("show_name")}

    perfold = []
    for variant in args.variants:
        path = FEAT.format(variant)
        if not os.path.exists(path):
            L.append(f"\n  [{variant}] {path} not found - skipped")
            continue
        fx = pd.read_csv(path)
        d = man.merge(fx.drop(columns=[c for c in ("dialect", "show_name", "duration")
                                       if c in fx.columns]),
                      on="wav_name", how="inner")
        fcols = [c for c in fx.columns if c not in META]
        X = np.nan_to_num(d[fcols].values.astype(float))
        labs = sorted(d.dialect.unique())
        lab2i = {l: i for i, l in enumerate(labs)}
        y = d.dialect.map(lab2i).values
        spk = d.speaker_id.astype(str).values
        shw = d.show_name.astype(str).values
        idx = np.arange(len(d))

        L.append(f"\n{'=' * 76}\n[{variant}]  n={len(d)}  features={len(fcols)}\n{'=' * 76}")
        res = {}
        for proto in ["random", "speaker", "show"]:
            rng = np.random.RandomState(SEED)
            pf, ty, tp = [], [], []
            for f in range(args.folds):
                if proto == "random":
                    te_pool = rng.choice(idx, size=N_TE, replace=False)
                    tr_pool = np.setdiff1d(idx, te_pool)
                elif proto == "speaker":
                    held = []
                    for s, spks in spk_by_show.items():
                        k = min(max(1, int(round(0.19 * len(spks)))), len(spks) - 1)
                        held.extend(rng.choice(spks, size=k, replace=False).tolist())
                    mask = np.isin(spk, held)
                    te_pool, tr_pool = idx[mask], idx[~mask]
                else:
                    held = [shows_by_lab[l][rng.randint(len(shows_by_lab[l]))]
                            for l in labs]
                    mask = np.isin(shw, held)
                    te_pool, tr_pool = idx[mask], idx[~mask]

                tr, te = match_sizes(tr_pool, te_pool, N_TR, N_TE, rng)
                if len(np.unique(y[tr])) < len(labs) or len(te) == 0:
                    continue
                p = fit_predict(X, y, tr, te, f)
                pf.append(macro_f1(y[te], p, list(range(len(labs)))))
                ty.append(y[te]); tp.append(p)
                perfold.append(dict(variant=variant, protocol=proto, fold=f,
                                    n_train=len(tr), n_test=len(te),
                                    macro_f1=pf[-1]))
            pooled = macro_f1(np.concatenate(ty), np.concatenate(tp),
                              list(range(len(labs))))
            res[proto] = (np.array(pf), pooled)
            L.append(f"  {proto:<9} mean-of-fold = {np.mean(pf):.3f} "
                     f"+/- {np.std(pf):.3f}   POOLED = {pooled:.3f}   (n={len(pf)})")

        if all(k in res for k in ("random", "speaker", "show")):
            from scipy import stats
            r, s, h = (res[k][0] for k in ("random", "speaker", "show"))
            n = min(len(r), len(s), len(h))
            d1 = r[:n] - s[:n]      # speaker memorisation
            d2 = s[:n] - h[:n]      # recording provenance, country constant
            L.append("")
            L.append(f"  random -> speaker (speaker memorisation) : {d1.mean():+.3f}")
            L.append(f"  speaker -> show   (recording provenance) : {d2.mean():+.3f}")
            t2, p2 = stats.ttest_rel(s[:n], h[:n])
            ci = stats.t.interval(0.95, n - 1, loc=d2.mean(),
                                  scale=d2.std(ddof=1) / np.sqrt(n))
            L.append(f"    speaker vs show: t={t2:.2f}  p={p2:.2e}  "
                     f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
                     f"wins {(d2 > 0).sum()}/{n}")
            if ci[0] > 0:
                L.append("    >>> holding out shows costs accuracy with COUNTRY CONSTANT.")
                L.append("        Country novelty cannot explain this drop.")
            else:
                L.append("    >>> no reliable drop: provenance leakage is not detectable")
                L.append("        here. AYDID would then be a low-leakage anchor.")

    L.append("\n" + "=" * 76)
    L.append("READING")
    L.append("=" * 76)
    L.append("  Compare the nonspeech row against full: if nonspeech retains most of")
    L.append("  the discrimination, the signal is not phonetic. Compare cmvn against")
    L.append("  full: the drop is the stationary channel component.")
    L.append("  Report ONE convention throughout the paper. Both are printed here so")
    L.append("  the choice is explicit and nothing is ever compared across them.")

    pd.DataFrame(perfold).to_csv(PERFOLD, index=False)
    rep = "\n".join(L)
    print("\n" + rep)
    open(REPORT, "w", encoding="utf-8").write(rep + "\n")
    print(f"\n[done] wrote {PERFOLD}")


if __name__ == "__main__":
    main()