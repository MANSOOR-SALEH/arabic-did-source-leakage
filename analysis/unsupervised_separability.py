#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unsupervised_separability.py
============================
Tests whether the paper's central practical claim is actually deliverable.

The manuscript states that the recommended measurement "requires only audio and
labels and no provenance metadata". That is false as the method is currently
written: Section 4.4 trains a classifier to predict SOURCE within each label,
which requires knowing each clip's source. An editor caught this immediately,
and it is the single most attractive claim in the paper.

So: can within-label acoustic structure be recovered WITHOUT source labels?

Within each label we compute, from the channel-only features alone:

  1. SILHOUETTE over k-means clusters, swept over k, taking the best k. High
     silhouette means the utterances of a single dialect fall into acoustically
     well-separated groups -- which is what having distinct recording sources
     looks like, without needing to know what those sources are.

  2. GAP-LIKE STATISTIC: silhouette on the real features minus mean silhouette
     on dimension-wise shuffled features. Shuffling destroys the covariance
     structure while preserving each marginal, so this controls for the trivial
     tendency of any high-dimensional data to cluster somewhat.

  3. EIGENVALUE CONCENTRATION: the share of variance in the leading principal
     components. Multiple distinct recording setups within one label produce a
     small number of dominant directions.

  4. NEAREST-NEIGHBOUR CONSISTENCY on a held-out speaker basis -- how often a
     clip's nearest neighbour shares its (unknown to the method) source. This
     one DOES peek at source labels, and is computed only as the validation
     target, never as part of the unsupervised score.

The test that matters is whether (1)-(3) rank the corpora the same way the
supervised separability does. If they do, the metadata-free claim is restored
and can be stated honestly. If they do not, the claim must be deleted from the
abstract and Section 6.

Reference points (supervised, raw macro-F1, from source_separability_v2):
    AYDID  programme  0.183      drop +0.018
    SADA_S programme  0.516      drop +0.275
    SADIC  condition  0.625      drop +0.212
    SADIC  country    0.716      drop +0.335

Usage:
    python unsupervised_separability.py
    python unsupervised_separability.py --variants full --kmax 10

Output: unsupervised_separability_report.txt, unsupervised_separability.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

CORPORA = {
    "aydid": dict(
        features="aydid_channel_features_{}.csv",
        label="dialect", source="show_name", speaker="speaker",
        meta=("wav_name", "speaker", "dialect", "show_name", "duration"),
        supervised=0.183, drop=0.018, unit="programme"),
    "sadic_country": dict(
        features="sadic16k_channel_features_{}.csv",
        label="region", source="country", speaker="speaker",
        meta=("wav_name", "speaker", "region", "country", "source",
              "domain", "duration"),
        supervised=0.716, drop=0.335, unit="country"),
    "sadic_condition": dict(
        features="sadic16k_channel_features_{}.csv",
        label="region", source="domain", speaker="speaker",
        meta=("wav_name", "speaker", "region", "country", "source",
              "domain", "duration"),
        supervised=0.625, drop=0.212, unit="condition"),
    "sada": dict(
        features="sada_channel_features_{}.csv",
        label="dialect", source="show_name", speaker="speaker",
        meta=("wav_name", "clip_id", "speaker", "dialect", "show_name",
              "environment", "duration"),
        supervised=0.516, drop=0.275, unit="programme"),
}

SEED = 0
MAX_N = 3000          # subsample per label, for tractable silhouette
N_SHUFFLE = 5


def keycol(df):
    return "wav_name" if "wav_name" in df.columns else "clip_id"


def unsupervised_scores(X, kmax, rng):
    """Silhouette (best k), shuffle-corrected silhouette, PCA concentration."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)

    def best_sil(M):
        best, bk = -1.0, None
        for k in range(2, kmax + 1):
            if len(M) < k * 5:
                break
            km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(M)
            if len(np.unique(km.labels_)) < 2:
                continue
            s = silhouette_score(M, km.labels_, sample_size=min(2000, len(M)),
                                 random_state=SEED)
            if s > best:
                best, bk = s, k
        return best, bk

    sil, k_best = best_sil(Z)

    # shuffle control: destroy covariance, keep marginals
    sh = []
    for _ in range(N_SHUFFLE):
        S = Z.copy()
        for j in range(S.shape[1]):
            rng.shuffle(S[:, j])
        s, _ = best_sil(S)
        sh.append(s)
    sil_null = float(np.mean(sh))

    p = PCA(n_components=min(10, Z.shape[1], Z.shape[0] - 1)).fit(Z)
    pc3 = float(p.explained_variance_ratio_[:3].sum())

    return dict(silhouette=float(sil), k_best=k_best,
                silhouette_null=sil_null, silhouette_gap=float(sil - sil_null),
                pca_top3=pc3)


def nn_source_consistency(X, src, rng, n_pairs=4000):
    """Validation only: how often is a clip's nearest neighbour same-source?
    Uses source labels, so it is NOT part of the unsupervised score."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    Z = StandardScaler().fit_transform(X)
    n = len(Z)
    idx = rng.choice(n, size=min(n_pairs, n), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    _, ind = nn.kneighbors(Z[idx])
    same = [src[idx[i]] == src[ind[i, 1]] for i in range(len(idx))]
    # chance = probability two random clips share a source
    _, cnt = np.unique(src, return_counts=True)
    pr = cnt / cnt.sum()
    return float(np.mean(same)), float(np.sum(pr ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["full"])
    ap.add_argument("--kmax", type=int, default=12)
    args = ap.parse_args()
    rng = np.random.RandomState(SEED)

    L = ["UNSUPERVISED WITHIN-LABEL ACOUSTIC STRUCTURE", "=" * 78,
         "  Can within-label source structure be detected WITHOUT source labels?",
         "  If yes, the paper's metadata-free claim is deliverable. If no, it must",
         "  be withdrawn from the abstract and Section 6.", ""]
    rows = []

    for name, cfg in CORPORA.items():
        for variant in args.variants:
            path = cfg["features"].format(variant)
            if not os.path.exists(path):
                L.append(f"  [{name}] {path} not found — skipped")
                continue
            df = pd.read_csv(path).dropna(
                subset=[cfg["label"], cfg["source"], cfg["speaker"]])
            meta = set(cfg["meta"]) | {cfg["label"], cfg["source"], cfg["speaker"]}
            fcols = [c for c in df.columns if c not in meta
                     and pd.api.types.is_numeric_dtype(df[c])]

            L.append(f"\n{'=' * 78}")
            L.append(f"[{name} / {variant}]  source unit = {cfg['unit']}  "
                     f"n={len(df)}  features={len(fcols)}")
            L.append("=" * 78)
            L.append(f"{'label':<14} {'n':>6} {'sil':>7} {'k':>3} {'sil_null':>9} "
                     f"{'gap':>7} {'PCA3':>7} {'NN-same':>8} {'chance':>7}")

            per = {"silhouette": [], "silhouette_gap": [], "pca_top3": [],
                   "nn": [], "nn_chance": []}
            for lab, g in df.groupby(cfg["label"]):
                if len(g) > MAX_N:
                    g = g.sample(MAX_N, random_state=SEED)
                X = np.nan_to_num(g[fcols].values.astype(float))
                src = g[cfg["source"]].astype(str).values
                if len(np.unique(src)) < 2 or len(g) < 60:
                    L.append(f"{str(lab):<14} {len(g):>6}   (too few sources/clips)")
                    continue
                s = unsupervised_scores(X, args.kmax, rng)
                nn, ch = nn_source_consistency(X, src, rng)
                L.append(f"{str(lab):<14} {len(g):>6} {s['silhouette']:>7.3f} "
                         f"{s['k_best']:>3} {s['silhouette_null']:>9.3f} "
                         f"{s['silhouette_gap']:>7.3f} {s['pca_top3']:>7.3f} "
                         f"{nn:>8.3f} {ch:>7.3f}")
                for k in ("silhouette", "silhouette_gap", "pca_top3"):
                    per[k].append(s[k])
                per["nn"].append(nn); per["nn_chance"].append(ch)

            if not per["silhouette"]:
                continue
            rec = dict(corpus=name, variant=variant, unit=cfg["unit"],
                       supervised=cfg["supervised"], drop=cfg["drop"],
                       silhouette=np.mean(per["silhouette"]),
                       sil_gap=np.mean(per["silhouette_gap"]),
                       pca_top3=np.mean(per["pca_top3"]),
                       nn_same=np.mean(per["nn"]),
                       nn_chance=np.mean(per["nn_chance"]))
            rec["nn_lift"] = rec["nn_same"] - rec["nn_chance"]
            rows.append(rec)
            L.append(f"\n  corpus means: silhouette={rec['silhouette']:.3f}  "
                     f"gap={rec['sil_gap']:.3f}  PCA3={rec['pca_top3']:.3f}  "
                     f"NN lift={rec['nn_lift']:+.3f}")
            L.append(f"  supervised separability = {cfg['supervised']:.3f}   "
                     f"measured drop = {cfg['drop']:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("unsupervised_separability.csv", index=False)

    L.append("\n" + "=" * 78)
    L.append("DOES ANY UNSUPERVISED SCORE RECOVER THE SUPERVISED ORDERING?")
    L.append("=" * 78)
    if len(out) >= 3:
        from scipy import stats
        L.append(f"{'corpus':<18} {'unit':<11} {'sil':>7} {'gap':>7} {'PCA3':>7} "
                 f"{'NNlift':>7} {'superv':>8} {'drop':>7}")
        for _, r in out.sort_values("drop").iterrows():
            L.append(f"{r.corpus:<18} {r.unit:<11} {r.silhouette:>7.3f} "
                     f"{r.sil_gap:>7.3f} {r.pca_top3:>7.3f} {r.nn_lift:>7.3f} "
                     f"{r.supervised:>8.3f} {r['drop']:>+7.3f}")
        L.append("")
        for col in ("silhouette", "sil_gap", "pca_top3", "nn_lift"):
            rs_sup, ps_sup = stats.spearmanr(out[col], out.supervised)
            rs_dr, ps_dr = stats.spearmanr(out[col], out['drop'])
            L.append(f"  {col:<12} vs supervised: rho={rs_sup:+.2f} (p={ps_sup:.2f})"
                     f"   vs drop: rho={rs_dr:+.2f} (p={ps_dr:.2f})")
        L.append("")
        L.append("  n = %d corpora. With four points a Spearman rho of 1.00 has" % len(out))
        L.append("  p = 0.083 at best, so no correlation here can reach conventional")
        L.append("  significance. Read the ORDERING, not the p-value, and report it")
        L.append("  as exploratory in the paper.")
        L.append("")
        best = max(("silhouette", "sil_gap", "pca_top3"),
                   key=lambda c: stats.spearmanr(out[c], out['drop'])[0])
        rho = stats.spearmanr(out[best], out['drop'])[0]
        if rho >= 0.8:
            L.append(f"  >>> '{best}' recovers the ordering (rho={rho:+.2f}) WITHOUT")
            L.append("      source labels. The metadata-free claim is deliverable:")
            L.append("      rewrite Section 4.4 around this statistic, report the")
            L.append("      supervised version as its validation, and keep the claim.")
        else:
            L.append(f"  >>> No unsupervised score tracks the drop (best rho={rho:+.2f}).")
            L.append("      DELETE the metadata-free claim from the abstract and from")
            L.append("      Section 6. State plainly that the measurement requires a")
            L.append("      source variable, which most corpora do not have — that is")
            L.append("      a limitation of the method, and reporting it honestly is")
            L.append("      better than a claim a reviewer disproves in one reading.")
        L.append("")
        L.append("  NN-lift uses source labels and is reported only as the validation")
        L.append("  target: it shows how much same-source structure exists at all.")
    else:
        L.append("  fewer than three corpora available — run the feature extractors first.")

    rep = "\n".join(L)
    print("\n" + rep)
    open("unsupervised_separability_report.txt", "w", encoding="utf-8").write(rep + "\n")
    print("\n[done] wrote unsupervised_separability.csv")


if __name__ == "__main__":
    main()
