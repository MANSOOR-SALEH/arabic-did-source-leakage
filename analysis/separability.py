#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_separability_v2.py
=========================
Tests the one explanation still standing.

Three candidate explanations for why corpora differ in leakage have now died:

  nesting              AYDID is nested and clean (+0.016); SADA_S is CROSSED
                       and leaks (+0.216).
  sources per label    flat across K=1..5 in AYDID; at K=1 source predicts the
                       label perfectly and the drop is still only +0.042.
  file format          re-encoding SADIC to uniform 16 kHz PCM changed the drop
                       by +0.000.

What remains: how ACOUSTICALLY DISTINCTIVE the sources are from one another.
SADIC's collections were gathered separately; SADA's shows are different
broadcasts; AYDID's 40 programmes were collected under similar conditions. If
separability tracks the measured drops, an acoustic measurement predicts
benchmark inflation where corpus structure does not -- and the probe becomes an
instrument rather than a diagnostic.

Measurement, identical in every corpus:

    WITHIN each label, predict SOURCE from the channel-only features,
    speaker-disjointly. The label is constant, so label identity cannot
    explain any separation found.

Reported against each label's own chance level, since the number of sources
per label differs.

Usage:
    python source_separability_v2.py
    python source_separability_v2.py --corpora sadic sada --variants full

Output: source_separability_v2_report.txt, source_separability_v2.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

# measured speaker -> source drops from the three-protocol runs
KNOWN_DROPS = {
    "aydid":        ("show holdout",      0.016),
    "sadic":        ("country holdout",   0.335),
    "sadic_domain": ("condition holdout", 0.196),
    "sada":         ("show holdout",      0.216),
}

CORPORA = {
    "aydid": dict(
        features="aydid_channel_features_{}.csv",
        label="dialect", source="show_name", speaker="speaker",
        meta=("wav_name", "speaker", "dialect", "show_name", "duration")),
    "sadic": dict(
        features="sadic16k_channel_features_{}.csv",
        label="region", source="country", speaker="speaker",
        meta=("wav_name", "speaker", "region", "country", "source",
              "domain", "duration")),
    "sadic_domain": dict(
        features="sadic16k_channel_features_{}.csv",
        label="region", source="domain", speaker="speaker",
        meta=("wav_name", "speaker", "region", "country", "source",
              "domain", "duration")),
    "sada": dict(
        features="sada_channel_features_{}.csv",
        label="dialect", source="show_name", speaker="speaker",
        meta=("wav_name", "speaker", "dialect", "show_name",
              "environment", "duration")),
}

SEED = 0
N_FOLDS = 10
MIN_PER_SOURCE = 20      # sources thinner than this are dropped


def macro_f1(y, p, ids):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=ids, average="macro", zero_division=0)


def separability(X, src, spk, n_folds, seed=SEED):
    """Speaker-disjoint source classification within one label."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    keep_src = pd.Series(src).value_counts()
    keep_src = set(keep_src[keep_src >= MIN_PER_SOURCE].index)
    m = np.isin(src, list(keep_src))
    if m.sum() < 50:
        return None
    X, src, spk = X[m], src[m], spk[m]

    srcs = sorted(pd.unique(src))
    if len(srcs) < 2:
        return None
    s2i = {s: i for i, s in enumerate(srcs)}
    y = np.array([s2i[s] for s in src])
    ids = list(range(len(srcs)))
    speakers = sorted(pd.unique(spk))
    if len(speakers) < 4:
        return None

    rng = np.random.RandomState(seed)
    scores = []
    for f in range(n_folds):
        sh = list(speakers); rng.shuffle(sh)
        hold = set(sh[:max(1, int(0.3 * len(sh)))])
        te = np.isin(spk, list(hold)); tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        mdl = LogisticRegression(max_iter=3000, random_state=f).fit(
            sc.transform(X[tr]), y[tr])
        scores.append(macro_f1(y[te], mdl.predict(sc.transform(X[te])), ids))
    if not scores:
        return None
    return dict(n_sources=len(srcs), chance=1.0 / len(srcs),
                mean=float(np.mean(scores)), sd=float(np.std(scores)),
                n_folds=len(scores), n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA))
    ap.add_argument("--variants", nargs="+", default=["full", "nonspeech"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    L = ["WITHIN-LABEL SOURCE SEPARABILITY", "=" * 78,
         "  Within each label, predict SOURCE from channel-only features,",
         "  speaker-disjointly. The label is constant, so label identity cannot",
         "  explain any separation found. Reported against each label's own",
         f"  chance level. Sources with fewer than {MIN_PER_SOURCE} clips dropped."]
    rows = []

    for corpus in args.corpora:
        cfg = CORPORA[corpus]
        for variant in args.variants:
            path = cfg["features"].format(variant)
            if not os.path.exists(path):
                L.append(f"\n  [{corpus}/{variant}] {path} not found — skipped")
                continue
            df = pd.read_csv(path)
            need = [cfg["label"], cfg["source"], cfg["speaker"]]
            if any(c not in df.columns for c in need):
                L.append(f"\n  [{corpus}/{variant}] missing {need} — skipped")
                continue
            df = df.dropna(subset=need)
            meta = set(cfg["meta"]) | set(need)
            fcols = [c for c in df.columns if c not in meta
                     and pd.api.types.is_numeric_dtype(df[c])]

            L.append(f"\n{'=' * 78}")
            L.append(f"[{corpus} / {variant}]  n={len(df)}  features={len(fcols)}  "
                     f"source={cfg['source']}")
            L.append("=" * 78)
            L.append(f"{'label':<14} {'srcs':>5} {'chance':>7} {'sep':>7} "
                     f"{'sd':>6} {'x chance':>9} {'n':>7}")

            per = []
            for lab, g in df.groupby(cfg["label"]):
                X = np.nan_to_num(g[fcols].values.astype(float))
                r = separability(X, g[cfg["source"]].astype(str).values,
                                 g[cfg["speaker"]].astype(str).values, args.folds)
                if r is None:
                    L.append(f"{str(lab):<14} {'—':>5}  (too few sources or speakers)")
                    continue
                ratio = r["mean"] / r["chance"]
                L.append(f"{str(lab):<14} {r['n_sources']:>5} {r['chance']:>7.3f} "
                         f"{r['mean']:>7.3f} {r['sd']:>6.3f} {ratio:>8.2f}x {r['n']:>7}")
                per.append(ratio)
                rows.append(dict(corpus=corpus, variant=variant, label=lab,
                                 n_sources=r["n_sources"], chance=r["chance"],
                                 separability=r["mean"], sd=r["sd"],
                                 above_chance=ratio, n=r["n"]))
            if per:
                L.append(f"\n  corpus mean separability = {np.mean(per):.2f}x chance")
                if corpus in KNOWN_DROPS:
                    what, dd = KNOWN_DROPS[corpus]
                    L.append(f"  measured drop ({what}) = {dd:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("source_separability_v2.csv", index=False)

    # ---------- the pairing ----------
    L.append("\n" + "=" * 78)
    L.append("DOES SEPARABILITY PREDICT THE DROP?")
    L.append("=" * 78)
    v0 = args.variants[0]
    pairs = []
    for corpus in args.corpora:
        g = out[(out.corpus == corpus) & (out.variant == v0)]
        if len(g) and corpus in KNOWN_DROPS:
            what, dd = KNOWN_DROPS[corpus]
            pairs.append((corpus, what, g.above_chance.mean(),
                          g.separability.mean(), dd))
    if pairs:
        L.append(f"{'corpus':<14} {'protocol':<18} {'sep(x chance)':>14} "
                 f"{'sep(raw)':>9} {'drop':>8}")
        for c, w, s, raw, dd in sorted(pairs, key=lambda t: t[4]):
            L.append(f"{c:<14} {w:<18} {s:>13.2f}x {raw:>9.3f} {dd:>+8.3f}")
        if len(pairs) >= 3:
            from scipy import stats
            x = np.array([p[2] for p in pairs]); yv = np.array([p[4] for p in pairs])
            r, p = stats.pearsonr(x, yv)
            rs, ps = stats.spearmanr(x, yv)
            L.append(f"\n  Pearson r  = {r:+.3f}  (p={p:.3f})")
            L.append(f"  Spearman r = {rs:+.3f}  (p={ps:.3f})   n={len(pairs)} corpora")
            L.append("")
            if rs > 0.8:
                L.append("  >>> Separability tracks the drop. An acoustic measurement")
                L.append("      predicts inflation where nesting, sources-per-label and")
                L.append("      file format all failed. This is the instrument claim:")
                L.append("      run the probe on any corpus, with no provenance metadata,")
                L.append("      and it tells you whether your speaker-disjoint number is")
                L.append("      trustworthy.")
            else:
                L.append("  >>> Separability does NOT track the drop either. Then the")
                L.append("      honest paper reports the four measurements and states")
                L.append("      plainly that no single corpus property explains them —")
                L.append("      which is still a useful warning, just a narrower one.")
            L.append("")
            L.append("  Caveat: correlation over a handful of corpora. State n and treat")
            L.append("  it as suggestive, not as a fitted calibration.")
    else:
        L.append("  not enough corpora with both measurements.")

    rep = "\n".join(L)
    print("\n" + rep)
    open("source_separability_v2_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done] wrote source_separability_v2.csv")


if __name__ == "__main__":
    main()
