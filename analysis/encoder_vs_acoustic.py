#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder_three_protocols_v3.py
=============================
Same as v2, with one fix: labels that have only ONE source no longer break the
source-disjoint arm.

The problem in v2: SADIC's EGY region maps to a single country, so "hold out one
source per label" never removed EGY, EGY was therefore absent from every test
set, and the class-coverage check rejected all 20 folds. The country arm
vanished entirely.

The fix, which is what the channel-feature runs already did: for a label with
one source, fall back to a finer grouping (SADIC's `source` column -- the three
Egyptian collections) and hold one of those out instead. Every label then
appears on both sides and the fold is evaluable.

Where no finer grouping exists, the label stays in training throughout and the
report says so explicitly, so the limitation is visible rather than silent.

Everything else is unchanged from v2: logistic regression on both arms,
identical folds, identical clips, stem-based key matching, both conventions
reported.

Usage:
    python encoder_three_protocols_v3.py --encoder wavlm
    python encoder_three_protocols_v3.py --encoder xlsr

Output: encoder_three_protocols_{enc}.txt / .csv
"""
import argparse
import os
import numpy as np
import pandas as pd

SADIC_MANIFEST = "sadic_manifest_16k.csv"
CHANNEL = {"aydid": "aydid_channel_features_full.csv",
           "sadic": "sadic16k_channel_features_full.csv",
           "sada": "sada_channel_features_full.csv"}
CHAN_META = ("wav_name", "clip_id", "speaker", "dialect", "region", "country",
             "source", "show_name", "environment", "domain", "duration")
N_FOLDS = 20

KNOWN = [("AYDID", "show holdout", 0.016),
         ("SADIC", "condition holdout", 0.196),
         ("SADIC", "country holdout", 0.335),
         ("SADA_S", "show holdout", 0.216)]


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]


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


def load_corpus(corpus, encoder, L):
    npz = f"enc_{corpus}_{encoder}.npz"
    if not os.path.exists(npz):
        L.append(f"  [{corpus}] {npz} missing — skipped")
        return None
    z = np.load(npz, allow_pickle=True)
    E = z["features"]
    d = pd.DataFrame({"key": [str(k) for k in z["keys"]],
                      "label": [str(v) for v in z["label"]],
                      "speaker": [str(v) for v in z["speaker"]]})
    if "TRUE_DOMAIN" in z:
        d["domain"] = [str(v) for v in z["TRUE_DOMAIN"]]
    d["stem"] = d["key"].apply(stem)
    d["_row"] = np.arange(len(d))

    if corpus == "sadic":
        man = pd.read_csv(SADIC_MANIFEST)
        man.columns = [c.strip() for c in man.columns]
        man["stem"] = man["wav_name"].apply(stem)
        cols = ["stem", "country"] + (["source"] if "source" in man.columns else [])
        d = d.merge(man[cols].drop_duplicates("stem"), on="stem", how="left")
        d["source"] = d["country"]
        # finer grouping for labels with a single country (EGY's collections)
        d["subsource"] = d.get("source_y", d.get("source"))
        if "source_y" in d.columns:
            d["subsource"] = d["source_y"]
        elif "source_x" in d.columns:
            d["subsource"] = d["source_x"]
    else:
        d["source"] = [str(v) for v in z["source"]]
        d["subsource"] = d["source"]

    ch = pd.read_csv(CHANNEL[corpus])
    keycol = "wav_name" if "wav_name" in ch.columns else "clip_id"
    ch["stem"] = ch[keycol].apply(stem)
    fc = [c for c in ch.columns if c not in CHAN_META and c != "stem"
          and pd.api.types.is_numeric_dtype(ch[c])]
    merged = d.merge(ch[["stem"] + fc].drop_duplicates("stem"),
                     on="stem", how="inner")
    merged = merged.dropna(subset=["label", "source", "speaker"])

    if len(merged) == 0:
        L.append(f"  [{corpus}] merge empty — key mismatch.")
        L.append(f"    npz key example : {d['key'].iloc[0]}")
        L.append(f"    channel {keycol} example : {ch[keycol].iloc[0]}")
        return None

    E = E[merged["_row"].values]
    C = np.nan_to_num(merged[fc].values.astype(float))
    L.append(f"  [{corpus}] encoder {E.shape}  channel {C.shape}  "
             f"labels={merged.label.nunique()} sources={merged.source.nunique()} "
             f"speakers={merged.speaker.nunique()}  (matched {len(merged)}/{len(d)})")
    return E, C, merged.reset_index(drop=True)


def make_folds(m, n_folds, L):
    src = m["source"].astype(str).values
    sub = m["subsource"].astype(str).values if "subsource" in m.columns else src
    spk = m["speaker"].astype(str).values
    lab = m["label"].astype(str).values
    n = len(m)
    idx = np.arange(n)
    out = {"random": [], "speaker": [], "source": []}

    srcs = sorted(pd.unique(src))
    spk_by_src = {s: sorted(pd.unique(spk[src == s])) for s in srcs}

    # per label: the pool of units that can be held out
    pools, fallback, stuck = {}, [], []
    for l in sorted(pd.unique(lab)):
        p = sorted(pd.unique(src[lab == l]))
        if len(p) > 1:
            pools[l] = ("source", p)
        else:
            q = sorted(pd.unique(sub[lab == l]))
            if len(q) > 1:
                pools[l] = ("subsource", q)
                fallback.append((l, len(q)))
            else:
                stuck.append(l)
    if fallback:
        for l, nq in fallback:
            L.append(f"    note: label {l} has one source; holding out one of its "
                     f"{nq} finer collections instead")
    if stuck:
        L.append(f"    note: labels {stuck} have a single source and no finer "
                 f"grouping — they stay in training throughout")

    for f in range(n_folds):
        r = np.random.RandomState(f)

        te = np.zeros(n, bool)
        te[r.choice(idx, size=n // 4, replace=False)] = True
        out["random"].append(te)

        held = []
        for s in srcs:
            sp = spk_by_src[s]
            if len(sp) < 2:
                continue
            k = min(max(1, int(round(0.25 * len(sp)))), len(sp) - 1)
            held.extend(r.choice(sp, size=k, replace=False).tolist())
        out["speaker"].append(np.isin(spk, held))

        mask = np.zeros(n, bool)
        for l, (kind, pool) in pools.items():
            pick = pool[r.randint(len(pool))]
            col = src if kind == "source" else sub
            mask |= (lab == l) & (col == pick)
        out["source"].append(mask)

    if "domain" in m.columns and m["domain"].nunique() > 1:
        dom = []
        for held in ["online", "spontaneous"]:
            te = (m["domain"].astype(str).values == held)
            if te.sum() and (~te).sum():
                dom.append(te)
        if dom:
            out["domain"] = dom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="wavlm")
    ap.add_argument("--corpora", nargs="+", default=["aydid", "sadic", "sada"])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()
    from scipy import stats

    L = [f"ENCODER THREE-PROTOCOL EVALUATION — {args.encoder.upper()}", "=" * 78,
         "  Logistic regression on BOTH arms, identical folds, identical clips.",
         "  The capacity comparison is therefore exact by construction.", ""]
    rows = []

    for corpus in args.corpora:
        got = load_corpus(corpus, args.encoder, L)
        if got is None:
            continue
        E, C, m = got
        labs = sorted(m.label.unique())
        l2i = {l: i for i, l in enumerate(labs)}
        y = m.label.map(l2i).values
        k = len(labs)

        L.append(f"\n{'=' * 78}\n[{corpus}]  {k} classes, chance {1/k:.3f}\n{'=' * 78}")
        folds = make_folds(m, args.folds, L)
        L.append(f"{'protocol':<10} {'repr':<8} {'mean-of-fold':>16} {'pooled':>9}")

        store = {}
        for proto, masks in folds.items():
            for name, X in (("encoder", E), ("channel", C)):
                pf, ty, tp, skipped = [], [], [], 0
                for f, te in enumerate(masks):
                    tr = ~te
                    if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < k \
                            or te.sum() == 0:
                        skipped += 1
                        continue
                    p = fit_predict(X, y, tr, te, f)
                    pf.append(macro_f1(y[te], p, k)); ty.append(y[te]); tp.append(p)
                if not pf:
                    L.append(f"{proto:<10} {name:<8}  no usable folds "
                             f"({skipped} rejected for class coverage)")
                    continue
                pooled = macro_f1(np.concatenate(ty), np.concatenate(tp), k)
                store[(proto, name)] = np.array(pf)
                note = f"  [{skipped} skipped]" if skipped else ""
                L.append(f"{proto:<10} {name:<8} {np.mean(pf):>10.3f} "
                         f"+/-{np.std(pf):<5.3f} {pooled:>9.3f}{note}")
                rows.append(dict(corpus=corpus, encoder=args.encoder,
                                 protocol=proto, repr=name, mean=np.mean(pf),
                                 sd=np.std(pf), pooled=pooled, n_folds=len(pf)))

        L.append("")
        for tgt in ["source", "domain"]:
            for name in ("encoder", "channel"):
                if ("speaker", name) not in store or (tgt, name) not in store:
                    continue
                a, b = store[("speaker", name)], store[(tgt, name)]
                nn = min(len(a), len(b))
                dd = a[:nn] - b[:nn]
                if nn > 2:
                    ci = stats.t.interval(0.95, nn - 1, loc=dd.mean(),
                                          scale=dd.std(ddof=1) / np.sqrt(nn))
                    t, p = stats.ttest_rel(a[:nn], b[:nn])
                    L.append(f"  speaker -> {tgt:<8} [{name:<7}] {dd.mean():+.3f}  "
                             f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.1e}  n={nn}")
                else:
                    L.append(f"  speaker -> {tgt:<8} [{name:<7}] {dd.mean():+.3f}  "
                             f"(n={nn}, too few folds for an interval)")

        for tgt in ["source", "domain"]:
            if (tgt, "encoder") in store and (tgt, "channel") in store:
                a, b = store[(tgt, "encoder")], store[(tgt, "channel")]
                nn = min(len(a), len(b))
                if nn < 3:
                    continue
                dd = a[:nn] - b[:nn]
                t, p = stats.ttest_rel(a[:nn], b[:nn])
                ci = stats.t.interval(0.95, nn - 1, loc=dd.mean(),
                                      scale=dd.std(ddof=1) / np.sqrt(nn))
                L.append(f"\n  encoder vs channel, {tgt}-disjoint : {dd.mean():+.3f}  "
                         f"CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.1e}  "
                         f"wins {(dd > 0).sum()}/{nn}")
                if ci[0] > 0:
                    L.append("  >>> encoder advantage survives exact capacity matching")
                elif ci[1] < 0:
                    L.append("  >>> channel features BEAT the encoder here")
                else:
                    L.append("  >>> no detectable difference")

    L.append("\n" + "=" * 78)
    L.append("SUMMARY — drops by corpus and representation")
    L.append("=" * 78)
    df = pd.DataFrame(rows)
    if len(df):
        for tgt in ["source", "domain"]:
            sub = df[df.protocol.isin(["speaker", tgt])]
            if sub.protocol.nunique() < 2:
                continue
            piv = sub.pivot_table(index=["corpus", "repr"], columns="protocol",
                                  values="mean")
            if {"speaker", tgt} <= set(piv.columns):
                piv["drop"] = piv["speaker"] - piv[tgt]
                L.append(f"\n  speaker -> {tgt}:")
                L.append("  " + piv.round(3).dropna().to_string().replace("\n", "\n  "))
        L.append("\n  Channel-only results already in hand:")
        for c, w, v in KNOWN:
            L.append(f"    {c:<8} {w:<18} {v:+.3f}")

    df.to_csv(f"encoder_three_protocols_{args.encoder}.csv", index=False)
    rep = "\n".join(L)
    print("\n" + rep)
    open(f"encoder_three_protocols_{args.encoder}.txt", "w",
         encoding="utf-8").write(rep + "\n")
    print("\n[done]")


if __name__ == "__main__":
    main()
