#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_density_sensitivity.py
=============================
Closes the last uncontrolled variable in the matched comparison.

Table 4 equalises class count, chance level, corpus size and clips per speaker,
but not the number of programmes each label has: SADA carries 16-26 and AYDID at
most seven. Every reviewer of the last round named this as an alternative
explanation for the contrast, and the earlier within-SADA sweep gave them reason
to \u2014 there the drop fell from +0.156 with one programme per label to +0.054
with five. If source density alone drives the difference, the cross-corpus
claim does not hold.

This forces both corpora to the same number of programmes per label and sweeps
that number. For each level K:

    AYDID  three dialects drawn at random from the seven that have >= K
           programmes; K programmes kept per dialect
    SADA   three dialects; K programmes kept per dialect, chosen from those
           carrying enough clips of that dialect

Both arms are then subsampled to the same clip count with whole speaker units
retained, and run through the identical speaker-to-source protocol.

Two design points matter. Restricting SADA to K programmes is itself a sampling
decision, and its programmes are acoustically heterogeneous, so a single subset
proves nothing: every cell is repeated over many draws and the distribution
reported, not a point estimate. And the comparison is only meaningful where both
corpora can supply K programmes for three labels, which caps K at AYDID's
minimum.

If SADA remains above AYDID at matched source counts, the cross-corpus contrast
survives its last confound. If it does not, source density was the mechanism and
the paper's conclusion must be weakened accordingly \u2014 that outcome is
reportable and is better established now.

Usage:
    python source_density_sensitivity.py
    python source_density_sensitivity.py --draws 30 --kmax 6

Output: source_density_report.txt, source_density.csv, fig_density.png
"""
import argparse
import os
import numpy as np
import pandas as pd

AYDID = "aydid_channel_features_full.csv"
SADA = "sada_channel_features_full.csv"
META = ("wav_name", "clip_id", "speaker", "dialect", "show_name",
        "environment", "duration", "stem")
MIN_CLIPS_SADA = 4          # speaker-unit density filter, as in the manuscript
MIN_CLIPS_PER_PROG = 10     # a programme must supply this many clips of a label
N_FOLDS = 20
TARGET_N = 1180


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


def increment(sub, folds=N_FOLDS):
    """speaker-disjoint -> source-disjoint increment on one subset."""
    fc = feats(sub)
    X = np.nan_to_num(sub[fc].values.astype(float))
    labs = sorted(sub.dialect.unique()); l2i = {l: i for i, l in enumerate(labs)}
    y = sub.dialect.map(l2i).values
    src = sub.show_name.astype(str).values
    spk = sub.speaker.astype(str).values
    k = len(labs); n = len(sub)
    src_by_lab = {l: sorted(np.unique(src[y == l2i[l]])) for l in labs}
    if any(len(v) < 2 for v in src_by_lab.values()):
        return None                      # cannot hold a source out
    srcs = sorted(np.unique(src))
    spk_by_src = {s: sorted(np.unique(spk[src == s])) for s in srcs}

    sp_scores, sr_scores = [], []
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
        for te, acc in ((te_s, sp_scores), (te_r, sr_scores)):
            tr = ~te
            if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k or te.sum() == 0:
                acc.append(np.nan); continue
            acc.append(macro_f1(y[te], fit_predict(X, y, tr, te, f), k))
    a, b = np.array(sp_scores), np.array(sr_scores)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return None
    d = a[ok] - b[ok]
    return dict(inc=float(d.mean()), wins=int((d > 0).sum()), folds=int(ok.sum()),
                spk=float(a[ok].mean()), src=float(b[ok].mean()), n=n)


def take_k_programmes(d, labs, K, rng, target):
    """Keep K programmes per label, then subsample to `target` clips keeping
    whole speaker units."""
    per = target // len(labs)
    parts = []
    for lab in labs:
        g = d[d.dialect == lab]
        counts = g.groupby("show_name").size()
        pool = [s for s, c in counts.items() if c >= MIN_CLIPS_PER_PROG]
        if len(pool) < K:
            return None
        keep = list(rng.choice(pool, K, replace=False))
        gg = g[g.show_name.isin(keep)]
        units = list(gg.speaker.unique()); rng.shuffle(units)
        sel, tot = [], 0
        for u in units:
            m = (gg.speaker == u).sum()
            if tot + m > per and sel:
                continue
            sel.append(u); tot += m
            if tot >= per:
                break
        sub = gg[gg.speaker.isin(sel)]
        if sub.show_name.nunique() < 2:
            return None                  # need >=2 programmes to hold one out
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--target", type=int, default=TARGET_N)
    args = ap.parse_args()

    L = ["SOURCE-DENSITY SENSITIVITY", "=" * 78,
         "  Both corpora forced to the same number of programmes per label.",
         "  Every cell repeated over independent draws; distributions reported."]

    if not (os.path.exists(AYDID) and os.path.exists(SADA)):
        L.append("  feature files missing"); print("\n".join(L)); return

    ay = pd.read_csv(AYDID).dropna(subset=["dialect", "show_name", "speaker"])
    sd = pd.read_csv(SADA).dropna(subset=["dialect", "show_name", "speaker"])
    cnt = sd.groupby("speaker").size()
    sd = sd[sd.speaker.isin(set(cnt[cnt >= MIN_CLIPS_SADA].index))].reset_index(drop=True)

    # how many programmes can each label actually supply?
    def pool_sizes(d):
        out = {}
        for lab, g in d.groupby("dialect"):
            c = g.groupby("show_name").size()
            out[lab] = int((c >= MIN_CLIPS_PER_PROG).sum())
        return out
    pa, ps = pool_sizes(ay), pool_sizes(sd)
    L.append(f"\n  programmes per label with >= {MIN_CLIPS_PER_PROG} clips")
    L.append(f"    AYDID  : {pa}")
    L.append(f"    SADA_S : {ps}")
    kmax = min(args.kmax, max(2, min(sorted(pa.values())[-3:])),
               max(2, min(sorted(ps.values())[-3:])))
    L.append(f"  usable range of K: 2 to {kmax}")
    if kmax < 2:
        L.append("  >>> no level of K is constructible in both corpora.")
        print("\n".join(L)); return

    rows = []
    L.append(f"\n{'K':>3} {'corpus':<8} {'draws':>6} {'n':>6} {'speaker':>8} "
             f"{'source':>8} {'increment':>18}")
    for K in range(2, kmax + 1):
        for name, d, pools in (("AYDID", ay, pa), ("SADA_S", sd, ps)):
            elig = [l for l, v in pools.items() if v >= K]
            if len(elig) < 3:
                L.append(f"{K:>3} {name:<8}  fewer than three labels have {K} programmes")
                continue
            incs, ns, sps, srs = [], [], [], []
            for i in range(args.draws):
                rng = np.random.RandomState(1000 * K + i +
                                            (0 if name == "AYDID" else 500))
                labs = list(rng.choice(elig, 3, replace=False))
                sub = take_k_programmes(d, labs, K, rng, args.target)
                if sub is None:
                    continue
                r = increment(sub, args.folds)
                if r:
                    incs.append(r["inc"]); ns.append(r["n"])
                    sps.append(r["spk"]); srs.append(r["src"])
            if len(incs) < 3:
                L.append(f"{K:>3} {name:<8}  too few usable draws")
                continue
            a = np.array(incs)
            L.append(f"{K:>3} {name:<8} {len(a):>6} {int(np.mean(ns)):>6} "
                     f"{np.mean(sps):>8.3f} {np.mean(srs):>8.3f} "
                     f"{a.mean():>+9.3f} \u00b1 {a.std():.3f}")
            rows.append(dict(K=K, corpus=name, draws=len(a), n=int(np.mean(ns)),
                             speaker=np.mean(sps), source=np.mean(srs),
                             inc=a.mean(), sd=a.std(),
                             lo=a.min(), hi=a.max()))

    if not rows:
        L.append("\n  nothing constructible"); print("\n".join(L)); return
    df = pd.DataFrame(rows)
    df.to_csv("source_density.csv", index=False)

    L.append("\n" + "=" * 78)
    L.append("DOES THE CONTRAST SURVIVE MATCHED SOURCE DENSITY?")
    L.append("=" * 78)
    verdict = []
    for K in sorted(df.K.unique()):
        a = df[(df.K == K) & (df.corpus == "AYDID")]
        b = df[(df.K == K) & (df.corpus == "SADA_S")]
        if len(a) and len(b):
            gap = float(b.inc.iloc[0] - a.inc.iloc[0])
            overlap = not (b.lo.iloc[0] > a.hi.iloc[0])
            L.append(f"  K={K}: SADA {b.inc.iloc[0]:+.3f}\u00b1{b.sd.iloc[0]:.3f} "
                     f"[{b.lo.iloc[0]:+.3f},{b.hi.iloc[0]:+.3f}]   "
                     f"AYDID {a.inc.iloc[0]:+.3f}\u00b1{a.sd.iloc[0]:.3f} "
                     f"[{a.lo.iloc[0]:+.3f},{a.hi.iloc[0]:+.3f}]   "
                     f"gap {gap:+.3f}   ranges "
                     f"{'OVERLAP' if overlap else 'disjoint'}")
            verdict.append((K, gap, overlap))
    L.append("")
    if verdict:
        gaps = [v[1] for v in verdict]
        anyov = any(v[2] for v in verdict)
        L.append(f"  mean gap across K: {np.mean(gaps):+.3f}  "
                 f"(unmatched-density figure in the manuscript: +0.220)")
        if np.mean(gaps) > 0.10 and not anyov:
            L.append("  >>> The contrast SURVIVES matched source density at every K, with")
            L.append("      disjoint ranges. Source count is not an alternative")
            L.append("      explanation, and Section 5.1.1 can say so. Add this as")
            L.append("      Section 5.1.2 and remove the caveat that source density")
            L.append("      could not be equalised.")
        elif np.mean(gaps) > 0.05:
            L.append("  >>> The contrast narrows but persists. Report both the matched")
            L.append("      and density-matched figures, and state that part of the")
            L.append("      original difference was source density.")
        else:
            L.append("  >>> The contrast largely disappears once source density is")
            L.append("      matched. Source count, not any intrinsic corpus property,")
            L.append("      accounts for the difference. The paper's cross-corpus")
            L.append("      conclusion must be weakened and this reported plainly.")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "figure.dpi": 200, "savefig.bbox": "tight"})
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for name, col, mk in (("SADA_S", "#D55E00", "s"), ("AYDID", "#009E73", "o")):
            g = df[df.corpus == name].sort_values("K")
            if not len(g):
                continue
            ax.errorbar(g.K, g.inc, yerr=g.sd, marker=mk, color=col, linewidth=1.6,
                        markersize=5.5, markeredgecolor="black",
                        markeredgewidth=0.5, capsize=3, label=name)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xlabel("programmes per label in both corpora (K)")
        ax.set_ylabel("speaker \u2192 source increment")
        ax.legend(frameon=False)
        ax.set_title("Contrast at matched source density", loc="left", pad=8)
        fig.savefig("fig_density.png"); fig.savefig("fig_density.pdf")
        L.append("\n  wrote fig_density.png")
    except Exception as e:
        L.append(f"\n  figure skipped ({type(e).__name__})")

    rep = "\n".join(L)
    print("\n" + rep)
    open("source_density_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done]")


if __name__ == "__main__":
    main()
