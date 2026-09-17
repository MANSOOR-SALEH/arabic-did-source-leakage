#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matched_comparison.py
=====================
The two analyses every reviewer of the last round asked for.

A1  LIKE-FOR-LIKE COMPARISON.
    The headline contrast currently sets the full seven-class AYDID corpus
    against a restricted, rebalanced three-class subset of SADA, so it confounds
    provenance leakage with class count, chance level and corpus size. This
    builds a matched AYDID subset -- three dialects, the same number of clips,
    the same chance level, comparable clips per speaker -- and runs the identical
    speaker-to-source protocol on both.

    One quantity cannot be matched and we do not pretend otherwise: SADA has
    16-26 programmes per label and AYDID at most seven, so source density
    differs by design. That is a property of the corpora, not of the analysis,
    and it is reported rather than hidden.

    Because the choice of which three AYDID dialects to keep is arbitrary, the
    comparison is repeated over several random draws of three dialects and the
    spread reported. A result that depends on which dialects were chosen is not
    a result.

A2  BOUNDING SADA'S RESIDUAL SPEAKER LEAKAGE.
    SADA provides no corpus-wide speaker identities, so its speaker unit is a
    file-and-speaker composite: one physical speaker appearing in two files
    counts as two units, and a unit-disjoint split need not be speaker-disjoint.
    Reviewers note that the reported increment may therefore retain a speaker
    component.

    We cannot recover true identities, but we can bound the effect. Holding out
    whole FILES guarantees that no file contributes to both sides, removing the
    within-file route by which a physical speaker could span the split. The gap
    between the unit-disjoint and file-disjoint increments is an estimate of how
    much of the increment that route could account for.

Both analyses use the acoustic features already extracted; no new extraction is
required. Runtime is a few minutes on CPU.

Usage:
    python matched_comparison.py
    python matched_comparison.py --draws 8 --folds 20

Output: matched_comparison_report.txt, matched_comparison.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

AYDID = "aydid_channel_features_full.csv"
SADA = "sada_channel_features_full.csv"
META = ("wav_name", "clip_id", "speaker", "dialect", "show_name",
        "environment", "duration", "stem")
N_FOLDS = 20
SEED = 0
MIN_CLIPS_SADA = 4          # as in the manuscript's restricted subset


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


def protocols(sub, lab_col, src_col, spk_col, folds, tag, L, group_col=None):
    """random / speaker(or group)-disjoint / source-disjoint on one subset."""
    fc = feats(sub)
    X = np.nan_to_num(sub[fc].values.astype(float))
    labs = sorted(sub[lab_col].unique()); l2i = {l: i for i, l in enumerate(labs)}
    y = sub[lab_col].map(l2i).values
    src = sub[src_col].astype(str).values
    spk = sub[(group_col or spk_col)].astype(str).values
    k = len(labs); n = len(sub); idx = np.arange(n)
    src_by_lab = {l: sorted(np.unique(src[y == l2i[l]])) for l in labs}
    if any(len(v) < 2 for v in src_by_lab.values()):
        L.append(f"    [{tag}] a label has a single source \u2014 not runnable")
        return None
    srcs = sorted(np.unique(src))
    spk_by_src = {s: sorted(np.unique(spk[src == s])) for s in srcs}

    res = {}
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
            L.append(f"    [{tag}] too few usable folds for {proto}")
            return None
        res[proto] = np.array(pf)

    r_, s_, h_ = res["random"], res["speaker"], res["source"]
    m = min(len(r_), len(s_), len(h_))
    inc = s_[:m] - h_[:m]
    L.append(f"    [{tag}] n={n} classes={k} chance={1/k:.3f} "
             f"sources={len(srcs)} units={sub[(group_col or spk_col)].nunique()}")
    L.append(f"    [{tag}] random={r_.mean():.3f} speaker={s_.mean():.3f} "
             f"source={h_.mean():.3f} (sd {h_.std():.3f})")
    L.append(f"    [{tag}] speaker\u2192source = {inc.mean():+.3f}  "
             f"({(inc>0).sum()}/{m} folds)   random\u2192source = "
             f"{(r_[:m]-h_[:m]).mean():+.3f}")
    return dict(n=n, k=k, inc=inc.mean(), wins=int((inc > 0).sum()), folds=m,
                rand=r_.mean(), spk=s_.mean(), src=h_.mean(), src_sd=h_.std(),
                n_sources=len(srcs))


# ------------------------------------------------------------------ A2
def a2(L, folds):
    L.append("\n" + "=" * 78)
    L.append("A2  BOUNDING SADA'S RESIDUAL SPEAKER LEAKAGE")
    L.append("=" * 78)
    L.append("  SADA's speaker unit is a file-and-speaker composite, so a")
    L.append("  unit-disjoint split need not be speaker-disjoint. Holding out whole")
    L.append("  FILES removes the within-file route by which one physical speaker")
    L.append("  could appear on both sides. The gap between the two increments")
    L.append("  bounds how much of the reported increment that route explains.")
    if not os.path.exists(SADA):
        L.append(f"  {SADA} missing \u2014 skipped"); return []
    d = pd.read_csv(SADA).dropna(subset=["dialect", "show_name", "speaker"])
    d["file"] = d["speaker"].astype(str).str.split(":").str[0]
    cnt = d.groupby("speaker").size()
    keep = set(cnt[cnt >= MIN_CLIPS_SADA].index)
    r = d[d.speaker.isin(keep)].reset_index(drop=True)
    L.append(f"\n  restricted subset: {len(r)} clips, {r.speaker.nunique()} units, "
             f"{r.file.nunique()} files, {r.show_name.nunique()} programmes")
    fpu = r.groupby("file")["speaker"].nunique()
    L.append(f"  units per file: median={fpu.median():.1f} max={fpu.max()}")

    rows = []
    a = protocols(r, "dialect", "show_name", "speaker", folds, "unit-disjoint", L)
    b = protocols(r, "dialect", "show_name", "speaker", folds, "file-disjoint", L,
                  group_col="file")
    if a and b:
        gap = a["inc"] - b["inc"]
        L.append(f"\n  unit-disjoint increment : {a['inc']:+.3f}")
        L.append(f"  file-disjoint increment : {b['inc']:+.3f}")
        L.append(f"  difference              : {gap:+.3f}")
        L.append("")
        if abs(gap) < 0.03:
            L.append("  >>> The two agree. The composite speaker unit accounts for little")
            L.append("      of the increment, and the caveat can be reported as bounded")
            L.append("      rather than open.")
        else:
            L.append("  >>> The stricter grouping changes the increment materially.")
            L.append("      Report the file-disjoint figure as the conservative estimate")
            L.append("      and state the gap as the residual-speaker bound.")
        rows = [dict(analysis="A2", arm="unit-disjoint", **a),
                dict(analysis="A2", arm="file-disjoint", **b)]
    return rows


# ------------------------------------------------------------------ A1
def a1(L, folds, draws):
    from itertools import combinations
    L.append("\n" + "=" * 78)
    L.append("A1  LIKE-FOR-LIKE COMPARISON")
    L.append("=" * 78)
    if not (os.path.exists(AYDID) and os.path.exists(SADA)):
        L.append("  feature files missing \u2014 skipped"); return []

    # --- SADA reference arm: restricted, class-balanced, as in the manuscript ---
    sd = pd.read_csv(SADA).dropna(subset=["dialect", "show_name", "speaker"])
    cnt = sd.groupby("speaker").size()
    sd = sd[sd.speaker.isin(set(cnt[cnt >= MIN_CLIPS_SADA].index))]
    rng = np.random.RandomState(SEED)
    per = sd.groupby("dialect").size().min()
    parts = []
    for lab, g in sd.groupby("dialect"):
        units = list(g.speaker.unique()); rng.shuffle(units)
        kept, tot = [], 0
        for u in units:
            mm = (g.speaker == u).sum()
            if tot + mm > per and kept:
                continue
            kept.append(u); tot += mm
            if tot >= per:
                break
        parts.append(g[g.speaker.isin(kept)])
    sada = pd.concat(parts, ignore_index=True)
    n_target = len(sada)
    cps_target = sada.groupby("speaker").size().median()
    L.append(f"\n  SADA reference arm: n={n_target}, 3 classes, chance 0.333, "
             f"{sada.show_name.nunique()} programmes, "
             f"median {cps_target:.0f} clips per unit")
    L.append("  matched on: class count, chance level, total clips, clips per unit")
    L.append("  NOT matched: sources per label (SADA 16\u201326, AYDID 4\u20137) \u2014 a")
    L.append("  property of the corpora that cannot be equalised.")
    r_sada = protocols(sada, "dialect", "show_name", "speaker", folds, "SADA  ", L)

    # --- AYDID matched arm, over several draws of three dialects ---
    ay = pd.read_csv(AYDID).dropna(subset=["dialect", "show_name", "speaker"])
    all_labs = sorted(ay.dialect.unique())
    combos = list(combinations(all_labs, 3))
    rng2 = np.random.RandomState(SEED)
    pick = [combos[i] for i in rng2.choice(len(combos), min(draws, len(combos)),
                                           replace=False)]
    L.append(f"\n  AYDID matched arms: {len(pick)} random draws of three dialects "
             f"from {len(all_labs)}")
    rows, incs = [], []
    for di, trio in enumerate(pick):
        g = ay[ay.dialect.isin(trio)]
        # subsample to the SADA size, keeping whole speakers, class-balanced
        r3 = np.random.RandomState(100 + di)
        want = n_target // 3
        ps = []
        for lab in trio:
            gg = g[g.dialect == lab]
            units = list(gg.speaker.unique()); r3.shuffle(units)
            kept, tot = [], 0
            for u in units:
                mm = min((gg.speaker == u).sum(), int(cps_target))
                if tot + mm > want and kept:
                    continue
                sel = gg[gg.speaker == u]
                if len(sel) > cps_target:
                    sel = sel.sample(int(cps_target), random_state=di)
                ps.append(sel); tot += len(sel)
                if tot >= want:
                    break
            _ = kept
        sub = pd.concat(ps, ignore_index=True)
        res = protocols(sub, "dialect", "show_name", "speaker", folds,
                        f"AYDID-{'/'.join(t[-2:] for t in trio)}", L)
        if res:
            incs.append(res["inc"])
            rows.append(dict(analysis="A1", arm="AYDID", trio="/".join(trio), **res))

    L.append("\n" + "-" * 78)
    if r_sada and incs:
        a = np.array(incs)
        L.append(f"  SADA  (matched reference) : {r_sada['inc']:+.3f}")
        L.append(f"  AYDID (matched, {len(a)} draws) : {a.mean():+.3f} "
                 f"+/- {a.std():.3f}   range [{a.min():+.3f}, {a.max():+.3f}]")
        diff = r_sada["inc"] - a.mean()
        L.append(f"  difference                : {diff:+.3f}")
        L.append("")
        L.append(f"  unmatched figures in the manuscript: AYDID +0.018, "
                 f"SADA +0.301, difference 0.283")
        L.append("")
        if diff > 0.15:
            L.append("  >>> The contrast SURVIVES matching on class count, chance level,")
            L.append("      corpus size and clips per speaker. The headline can be")
            L.append("      restated on the matched comparison, which is what the")
            L.append("      reviewers asked for, with source density noted as the one")
            L.append("      quantity that could not be equalised.")
        elif diff > 0.05:
            L.append("  >>> The contrast shrinks under matching but remains substantial.")
            L.append("      Report the matched figure as the headline and the unmatched")
            L.append("      one as the upper bound.")
        else:
            L.append("  >>> The contrast largely disappears under matching. The original")
            L.append("      difference was driven by class count, chance level or corpus")
            L.append("      size rather than by provenance, and the paper's headline")
            L.append("      must be rewritten around the matched result.")
        if a.std() > 0.05:
            L.append("")
            L.append("  !! The AYDID increment varies substantially with which three")
            L.append("     dialects are drawn; report the spread, not a single value.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--draws", type=int, default=6)
    args = ap.parse_args()

    L = ["MATCHED COMPARISON AND SPEAKER-UNIT BOUND", "=" * 78,
         "  Uses the acoustic features already extracted. No new extraction."]
    rows = a1(L, args.folds, args.draws)
    rows += a2(L, args.folds)

    if rows:
        pd.DataFrame(rows).to_csv("matched_comparison.csv", index=False)
    rep = "\n".join(L)
    print("\n" + rep)
    open("matched_comparison_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done] wrote matched_comparison.csv")


if __name__ == "__main__":
    main()
