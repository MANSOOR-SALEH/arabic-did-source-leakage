#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sada_matched_draws.py
=====================
Removes an asymmetry in the matched comparison.

The AYDID arm of Table 4 is averaged over six random draws of three dialects,
because which three dialects are kept is an arbitrary choice. The SADA arm is a
single draw, justified in the manuscript on the grounds that SADA has exactly
three dialects and so there is no choice to be made.

That justification answers the wrong question. The dialects are not chosen, but
WHICH SPEAKER UNITS SURVIVE class rebalancing is, and that choice is just as
arbitrary as the choice of dialects. A reviewer has asked whether the SADA
subset is one of the five rebalanced subsamples of Section 3.2 or a separate
draw; it is a separate draw, and the right response is not to explain the
asymmetry but to remove it.

This repeats the SADA matched arm over several independent rebalancing draws
and reports the spread, so that both arms of Table 4 carry one.

If the spread is small the table gains a plus-or-minus and the objection
disappears. If it is large, the single-draw figure in the manuscript was not
representative and the headline needs revising \u2014 better established now.

Usage:
    python sada_matched_draws.py
    python sada_matched_draws.py --draws 8 --folds 20

Output: sada_matched_draws_report.txt
"""
import argparse
import os
import numpy as np
import pandas as pd

SADA = "sada_channel_features_full.csv"
META = ("wav_name", "clip_id", "speaker", "dialect", "show_name",
        "environment", "duration", "stem")
MIN_CLIPS = 4
N_FOLDS = 20
TARGET_N = 1182          # the matched arm size reported in Table 4


def feats(df):
    return [c for c in df.columns if c not in META
            and pd.api.types.is_numeric_dtype(df[c])]


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


def build_matched(d, rng, target):
    """Class-balanced subset of the restricted set, subsampled to `target`
    clips, dropping whole speaker units so speaker structure is preserved."""
    per = target // d.dialect.nunique()
    parts = []
    for lab, g in d.groupby("dialect"):
        units = list(g.speaker.unique())
        rng.shuffle(units)
        kept, tot = [], 0
        for u in units:
            m = (g.speaker == u).sum()
            if tot + m > per and kept:
                continue
            kept.append(u); tot += m
            if tot >= per:
                break
        parts.append(g[g.speaker.isin(kept)])
    return pd.concat(parts, ignore_index=True)


def protocols(sub, folds):
    fc = feats(sub)
    X = np.nan_to_num(sub[fc].values.astype(float))
    labs = sorted(sub.dialect.unique()); l2i = {l: i for i, l in enumerate(labs)}
    y = sub.dialect.map(l2i).values
    src = sub.show_name.astype(str).values
    spk = sub.speaker.astype(str).values
    k = len(labs); n = len(sub); idx = np.arange(n)
    src_by_lab = {l: sorted(np.unique(src[y == l2i[l]])) for l in labs}
    if any(len(v) < 2 for v in src_by_lab.values()):
        return None
    srcs = sorted(np.unique(src))
    spk_by_src = {s: sorted(np.unique(spk[src == s])) for s in srcs}

    out = {}
    for proto in ("random", "speaker", "source"):
        pf = []
        for f in range(folds):
            r = np.random.RandomState(f)
            if proto == "random":
                te = np.zeros(n, bool)
                te[r.choice(idx, size=max(1, n // 4), replace=False)] = True
            elif proto == "speaker":
                held = []
                for s in srcs:
                    sp = spk_by_src[s]
                    if len(sp) < 2:
                        continue
                    kk = min(max(1, int(round(0.25 * len(sp)))), len(sp) - 1)
                    held.extend(r.choice(sp, size=kk, replace=False).tolist())
                te = np.isin(spk, held)
            else:
                held = [pool[r.randint(len(pool))] for pool in src_by_lab.values()]
                te = np.isin(src, held)
            tr = ~te
            if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k or te.sum() == 0:
                continue
            pf.append(macro_f1(y[te], fit_predict(X, y, tr, te, f), k))
        if len(pf) < 3:
            return None
        out[proto] = np.array(pf)
    r_, s_, h_ = out["random"], out["speaker"], out["source"]
    m = min(len(r_), len(s_), len(h_))
    inc = s_[:m] - h_[:m]
    return dict(n=n, rand=r_.mean(), spk=s_.mean(), src=h_.mean(),
                inc=inc.mean(), wins=int((inc > 0).sum()), folds=m,
                units=sub.speaker.nunique(), shows=sub.show_name.nunique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--target", type=int, default=TARGET_N)
    args = ap.parse_args()

    L = ["SADA MATCHED ARM OVER INDEPENDENT DRAWS", "=" * 78,
         "  Table 4's AYDID arm averages six draws; its SADA arm is a single draw.",
         "  Which speaker units survive rebalancing is as arbitrary as which three",
         "  dialects are kept, so both arms should carry a spread. This repeats the",
         "  SADA arm over independent rebalancing draws."]

    if not os.path.exists(SADA):
        L.append(f"  {SADA} missing"); print("\n".join(L)); return
    d = pd.read_csv(SADA).dropna(subset=["dialect", "show_name", "speaker"])
    cnt = d.groupby("speaker").size()
    d = d[d.speaker.isin(set(cnt[cnt >= MIN_CLIPS].index))].reset_index(drop=True)
    L.append(f"\n  restricted set: {len(d)} clips, {d.speaker.nunique()} units, "
             f"{d.show_name.nunique()} programmes")
    L.append(f"  target matched size: {args.target} clips")
    L.append(f"\n{'draw':>5} {'n':>6} {'units':>6} {'shows':>6} {'random':>8} "
             f"{'speaker':>8} {'source':>8} {'incr':>8} {'folds':>7}")

    rows = []
    for i in range(args.draws):
        rng = np.random.RandomState(200 + i)
        sub = build_matched(d, rng, args.target)
        r = protocols(sub, args.folds)
        if r is None:
            L.append(f"{i:>5}  not runnable (a label lost its second programme)")
            continue
        L.append(f"{i:>5} {r['n']:>6} {r['units']:>6} {r['shows']:>6} "
                 f"{r['rand']:>8.3f} {r['spk']:>8.3f} {r['src']:>8.3f} "
                 f"{r['inc']:>+8.3f} {r['wins']:>4}/{r['folds']}")
        rows.append(r)

    if not rows:
        L.append("\n  no usable draws"); print("\n".join(L)); return

    inc = np.array([r["inc"] for r in rows])
    L.append("\n" + "=" * 78)
    L.append("SUMMARY")
    L.append("=" * 78)
    L.append(f"  SADA matched increment : {inc.mean():+.3f} +/- {inc.std():.3f} "
             f"over {len(inc)} draws")
    L.append(f"  range                  : [{inc.min():+.3f}, {inc.max():+.3f}]")
    L.append(f"  mean absolute scores   : random {np.mean([r['rand'] for r in rows]):.3f}, "
             f"speaker {np.mean([r['spk'] for r in rows]):.3f}, "
             f"source {np.mean([r['src'] for r in rows]):.3f}")
    L.append(f"  mean n                 : {int(np.mean([r['n'] for r in rows]))} clips")
    L.append("")
    L.append("  manuscript reports, from a single draw : +0.383")
    L.append("  AYDID matched arm, six draws           : +0.062 +/- 0.023")
    L.append(f"  matched difference using the mean      : "
             f"{inc.mean() - 0.062:+.3f}  (manuscript: +0.321)")
    L.append("")
    if inc.std() < 0.04 and abs(inc.mean() - 0.383) < 0.05:
        L.append("  >>> The single draw was representative. Report the mean and spread")
        L.append("      in Table 4 so both arms carry one, and the asymmetry the")
        L.append("      reviewer identified disappears.")
    elif inc.std() >= 0.04:
        L.append("  >>> The increment varies substantially with the rebalancing draw.")
        L.append("      The single-draw figure was not representative; report the mean")
        L.append("      and the range, and say that the estimate is draw-sensitive.")
    else:
        L.append("  >>> The mean differs materially from the single draw reported.")
        L.append("      Replace the manuscript figure with the mean over draws.")

    rep = "\n".join(L)
    print("\n" + rep)
    open("sada_matched_draws_report.txt", "w", encoding="utf-8").write(rep + "\n")
    pd.DataFrame(rows).to_csv("sada_matched_draws.csv", index=False)
    print("\n[done]")


if __name__ == "__main__":
    main()
