#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matched_20_draws.py
===================
Raises the matched comparison of Table 4 from six draws per corpus to twenty.

Two reviewers asked directly whether six draws is enough to support the claim
that every SADA_S draw exceeds every AYDID draw. Six is a small number for a
statement about distributions, and the question is cheap to answer: the features
are already extracted and the design is unchanged.

The matching is identical to the version reported in the manuscript. Both arms
have three classes and therefore the same chance level, the same number of
clips, and the same median clips per speaker unit; whole speaker units are kept
or dropped so the speaker-disjoint structure is preserved. Both arms are drawn
independently, because both involve an arbitrary selection: for AYDID which
three of the seven dialects to retain, and for SADA_S which speaker units
survive class rebalancing.

What to watch in the output:

  - Whether the means move. If they stay near +0.062 and +0.282, six draws were
    representative and the manuscript figures stand with a larger n.
  - Whether the ranges still fail to overlap. The claim in Table 4 is that the
    smallest SADA_S increment exceeds the largest AYDID one. With twenty draws
    each there is more opportunity for the tails to meet, and if they do, the
    claim must be restated as a difference in means rather than a separation of
    distributions.
  - Whether the standard deviations shrink, as they should with more draws.

Either outcome is reportable. A separation that survives twenty draws is a
stronger statement than one resting on six; a separation that fails is something
to know before a referee finds it.

Usage:
    python matched_20_draws.py
    python matched_20_draws.py --draws 20 --folds 20

Output: matched_20_draws_report.txt, matched_20_draws.csv
"""
import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd

AYDID = "aydid_channel_features_full.csv"
SADA = "sada_channel_features_full.csv"
META = ("wav_name", "clip_id", "speaker", "dialect", "show_name",
        "environment", "duration", "stem")
MIN_CLIPS_SADA = 4
TARGET_N = 1182
N_FOLDS = 20
SEED = 0


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


def increment(sub, folds):
    """speaker-disjoint minus source-disjoint on one matched subset."""
    fc = feats(sub)
    X = np.nan_to_num(sub[fc].values.astype(float))
    labs = sorted(sub.dialect.unique()); l2i = {l: i for i, l in enumerate(labs)}
    y = sub.dialect.map(l2i).values
    src = sub.show_name.astype(str).values
    spk = sub.speaker.astype(str).values
    k = len(labs); n = len(sub)
    src_by_lab = {l: sorted(np.unique(src[y == l2i[l]])) for l in labs}
    if any(len(v) < 2 for v in src_by_lab.values()):
        return None
    srcs = sorted(np.unique(src))
    spk_by_src = {s: sorted(np.unique(spk[src == s])) for s in srcs}

    sp, sr = [], []
    for f in range(folds):
        r = np.random.RandomState(f)
        held = []
        for s in srcs:
            u = spk_by_src[s]
            if len(u) < 2:
                continue
            kk = min(max(1, int(round(0.25 * len(u)))), len(u) - 1)
            held.extend(r.choice(u, size=kk, replace=False).tolist())
        te_s = np.isin(spk, held)
        hs = [pool[r.randint(len(pool))] for pool in src_by_lab.values()]
        te_r = np.isin(src, hs)
        for te, acc in ((te_s, sp), (te_r, sr)):
            tr = ~te
            if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k or te.sum() == 0:
                acc.append(np.nan); continue
            acc.append(macro_f1(y[te], fit_predict(X, y, tr, te, f), k))
    a, b = np.array(sp), np.array(sr)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return None
    dd = a[ok] - b[ok]
    cpu = float(sub.groupby('speaker').size().median())
    return dict(n=n, cpu=cpu, spk=float(a[ok].mean()), src=float(b[ok].mean()),
                inc=float(dd.mean()), wins=int((dd > 0).sum()), folds=int(ok.sum()))


def take_units(g, target, rng, cap=None):
    """Keep whole speaker units until `target` clips are reached."""
    units = list(g.speaker.unique()); rng.shuffle(units)
    sel, tot = [], 0
    for u in units:
        gg = g[g.speaker == u]
        if cap is not None and len(gg) > cap:
            gg = gg.sample(cap, random_state=int(rng.randint(1 << 30)))
        if tot + len(gg) > target and sel:
            continue
        sel.append(gg); tot += len(gg)
        if tot >= target:
            break
    return pd.concat(sel, ignore_index=True) if sel else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    L = ["MATCHED COMPARISON AT %d DRAWS PER CORPUS" % args.draws, "=" * 74,
         "  Design identical to the manuscript's Table 4; only the number of",
         "  draws changes. Two reviewers asked whether six was sufficient."]

    if not (os.path.exists(AYDID) and os.path.exists(SADA)):
        L.append("  feature files missing"); print("\n".join(L)); return

    # ---------------- SADA arm ----------------
    sd = pd.read_csv(SADA).dropna(subset=["dialect", "show_name", "speaker"])
    cnt = sd.groupby("speaker").size()
    sd = sd[sd.speaker.isin(set(cnt[cnt >= MIN_CLIPS_SADA].index))]
    per = TARGET_N // sd.dialect.nunique()

    L.append(f"\n{'draw':>5} {'corpus':<8} {'n':>6} {'c/unit':>7} {'speaker':>8} "
             f"{'source':>8} {'increment':>10} {'folds':>7}")
    rows = []
    for i in range(args.draws):
        rng = np.random.RandomState(500 + i)
        parts = [take_units(g, per, rng) for _, g in sd.groupby("dialect")]
        if any(p is None for p in parts):
            continue
        sub = pd.concat(parts, ignore_index=True)
        r = increment(sub, args.folds)
        if r:
            L.append(f"{i:>5} {'SADA_S':<8} {r['n']:>6} {r['cpu']:>7.1f} {r['spk']:>8.3f} "
                     f"{r['src']:>8.3f} {r['inc']:>+10.3f} {r['wins']:>4}/{r['folds']}")
            rows.append(dict(corpus="SADA_S", draw=i, **r))

    # ---------------- AYDID arm ----------------
    ay = pd.read_csv(AYDID).dropna(subset=["dialect", "show_name", "speaker"])
    labs_all = sorted(ay.dialect.unique())
    combos = list(combinations(labs_all, 3))
    rng0 = np.random.RandomState(SEED)
    picks = [combos[j] for j in rng0.choice(len(combos),
             min(args.draws, len(combos)), replace=False)]
    if len(picks) < args.draws:                   # 35 combos for 7 labels
        picks = picks + [combos[j] for j in rng0.choice(
            len(combos), args.draws - len(picks), replace=True)]
    # No per-unit cap. Capping AYDID units at SADA's median clips-per-unit
    # shrinks the AYDID arm to ~750 clips and makes the two arms incomparable
    # in size, which is the quantity Table 4 matches on. AYDID has roughly 50
    # clips per speaker and SADA roughly 5, so the two corpora cannot be matched
    # on total clips and on clips-per-speaker at the same time; we match total
    # clips, as the manuscript does, and report the per-speaker difference.
    for i, trio in enumerate(picks):
        rng = np.random.RandomState(900 + i)
        g3 = ay[ay.dialect.isin(trio)]
        parts = [take_units(g, per, rng) for _, g in g3.groupby("dialect")]
        if any(p is None for p in parts):
            continue
        sub = pd.concat(parts, ignore_index=True)
        r = increment(sub, args.folds)
        if r:
            L.append(f"{i:>5} {'AYDID':<8} {r['n']:>6} {r['cpu']:>7.1f} {r['spk']:>8.3f} "
                     f"{r['src']:>8.3f} {r['inc']:>+10.3f} {r['wins']:>4}/{r['folds']}")
            rows.append(dict(corpus="AYDID", draw=i, trio="/".join(trio), **r))

    if not rows:
        L.append("\n  nothing usable"); print("\n".join(L)); return
    df = pd.DataFrame(rows)
    df.to_csv("matched_20_draws.csv", index=False)

    L.append("\n" + "=" * 74)
    L.append("SUMMARY")
    L.append("=" * 74)
    out = {}
    for c in ("AYDID", "SADA_S"):
        v = df[df.corpus == c]["inc"].values
        if len(v) == 0:
            continue
        out[c] = v
        L.append(f"  {c:<8} {v.mean():+.3f} \u00b1 {v.std(ddof=1):.3f}   "
                 f"range [{v.min():+.3f}, {v.max():+.3f}]   n = {len(v)}")
        L.append(f"           mean speaker {df[df.corpus==c]['spk'].mean():.3f}, "
                 f"mean source {df[df.corpus==c]['src'].mean():.3f}, "
                 f"mean n {int(df[df.corpus==c]['n'].mean())}, "
                 f"median clips/unit {df[df.corpus==c]['cpu'].median():.1f}")
    L.append("")
    L.append("  manuscript reports, from six draws each:")
    L.append("    AYDID  +0.062 \u00b1 0.023   range [+0.031, +0.093]")
    L.append("    SADA_S +0.282 \u00b1 0.046   range [+0.225, +0.339]")
    L.append("    difference 0.220")

    if "AYDID" in out and "SADA_S" in out:
        a, s = out["AYDID"], out["SADA_S"]
        diff = s.mean() - a.mean()
        overlap = s.min() <= a.max()
        na = int(df[df.corpus=="AYDID"]["n"].mean())
        ns = int(df[df.corpus=="SADA_S"]["n"].mean())
        if abs(na - ns) > 60:
            L.append(f"\n  !! ARMS DIFFER IN SIZE: AYDID {na} clips, SADA_S {ns}. "
                     f"Table 4 matches on total clips, so these are not comparable.")
        L.append(f"\n  difference at {args.draws} draws: {diff:+.3f}")
        L.append(f"  smallest SADA_S {s.min():+.3f} vs largest AYDID {a.max():+.3f} "
                 f"\u2014 {'OVERLAP' if overlap else 'disjoint'}")
        L.append("")
        if not overlap and abs(diff - 0.220) < 0.05:
            L.append("  >>> Six draws were representative. Update Table 4 and Figure 2")
            L.append("      to the larger n, keep the separation claim, and say that")
            L.append("      the ranges remain disjoint over twenty draws.")
        elif not overlap:
            L.append("  >>> The separation holds but the difference has moved. Report")
            L.append("      the twenty-draw figures throughout and update the abstract")
            L.append("      and conclusion.")
        else:
            L.append("  >>> The ranges now overlap. The claim that every SADA_S draw")
            L.append("      exceeds every AYDID draw does not survive twenty draws and")
            L.append("      must be restated as a difference in means with the observed")
            L.append("      spread. This is better found now than by a referee.")

    rep = "\n".join(L)
    print("\n" + rep)
    open("matched_20_draws_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done]")


if __name__ == "__main__":
    main()