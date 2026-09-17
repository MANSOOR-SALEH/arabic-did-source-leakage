#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_color.py
=====================
Colour version of the three manuscript figures.

Palette is Okabe-Ito (colourblind-safe), which also retains reasonable
separation if a reviewer prints in greyscale.

Output: fig1_drops, fig2_separability, fig3_protocols  (.pdf and .png)

Usage:
    python make_figures_color.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Okabe-Ito colourblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERM   = "#D55E00"
PURPLE = "#CC79A7"
SKY    = "#56B4E9"

REPR_COLORS = [BLUE, ORANGE, GREEN]          # channel, WavLM, XLS-R
CORPUS_COLORS = {"AYDID": GREEN, "SADA_S": VERM, "SADIC": BLUE}

FALLBACK = {
    ("AYDID",  "programme holdout", "channel"): (0.018, -0.003, 0.038),
    ("AYDID",  "programme holdout", "WavLM"):   (0.040,  0.007, 0.074),
    ("AYDID",  "programme holdout", "XLS-R"):   (0.039,  0.006, 0.071),
    ("SADA_S", "programme holdout", "channel"): (0.275,  0.212, 0.338),
    ("SADA_S", "programme holdout", "WavLM"):   (0.267,  0.229, 0.305),
    ("SADA_S", "programme holdout", "XLS-R"):   (0.270,  0.229, 0.311),
    ("SADIC",  "condition holdout", "channel"): (0.212, np.nan, np.nan),
    ("SADIC",  "condition holdout", "WavLM"):   (0.209, np.nan, np.nan),
    ("SADIC",  "condition holdout", "XLS-R"):   (0.197, np.nan, np.nan),
    ("SADIC",  "country holdout",   "channel"): (0.335,  0.289, 0.380),
}

PROTO_ABS = {
    "AYDID  (7-way)": {"random": 0.641, "speaker": 0.593, "source": 0.576},
    "SADA_S (3-way)": {"random": 0.523, "speaker": 0.509, "source": 0.234},
    "SADIC  (4-way)": {"random": 0.735, "speaker": 0.694, "source": 0.482},
}
CHANCE = {"AYDID  (7-way)": 1/7, "SADA_S (3-way)": 1/3, "SADIC  (4-way)": 1/4}
LINE_COLORS = {"AYDID  (7-way)": GREEN, "SADA_S (3-way)": VERM,
               "SADIC  (4-way)": BLUE}

SEP = [
    ("AYDID",  "programme", 0.183, 0.018, 0.178),
    ("SADA_S", "programme", 0.516, 0.275, 0.047),
    ("SADIC",  "condition", 0.625, 0.212, 0.349),
    ("SADIC",  "country",   0.716, 0.335, 0.204),
]


def load_encoder_csvs():
    out = {}
    for enc, tag in (("wavlm", "WavLM"), ("xlsr", "XLS-R")):
        path = f"encoder_three_protocols_{enc}.csv"
        if not os.path.exists(path):
            print(f"  note: {path} not found — using declared values")
            continue
        df = pd.read_csv(path)
        for corpus, name in (("aydid", "AYDID"), ("sada", "SADA_S")):
            g = df[df.corpus == corpus]
            sp, sr = g[g.protocol == "speaker"], g[g.protocol == "source"]
            for repr_, label in (("encoder", tag), ("channel", "channel")):
                a = sp[sp["repr"] == repr_]["mean"]
                b = sr[sr["repr"] == repr_]["mean"]
                if len(a) and len(b):
                    out[(name, "programme holdout", label)] = (
                        float(a.iloc[0] - b.iloc[0]), np.nan, np.nan)
    return out


def figure1(drops):
    cells = [("AYDID", "programme holdout"), ("SADA_S", "programme holdout"),
             ("SADIC", "condition holdout"), ("SADIC", "country holdout")]
    reprs = ["channel", "WavLM", "XLS-R"]

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    w, x = 0.26, np.arange(len(cells))

    for i, rp in enumerate(reprs):
        vals, los, his = [], [], []
        for c in cells:
            v = drops.get((c[0], c[1], rp), (np.nan, np.nan, np.nan))
            vals.append(v[0])
            los.append(v[0] - v[1] if not np.isnan(v[1]) else np.nan)
            his.append(v[2] - v[0] if not np.isnan(v[2]) else np.nan)
        err = np.array([los, his])
        ax.bar(x + (i - 1) * w, vals, w, label=rp, color=REPR_COLORS[i],
               edgecolor="black", linewidth=0.5, alpha=0.92,
               yerr=np.where(np.isnan(err), 0, err), capsize=2.5,
               error_kw={"linewidth": 0.8, "ecolor": "#333333"})
        tops = [v if np.isnan(h) else v + h for v, h in zip(vals, his)]
        for xi, v, t in zip(x + (i - 1) * w, vals, tops):
            if not np.isnan(v):
                ax.text(xi, t + 0.010, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c[0]}\n{c[1].replace(' holdout','')}" for c in cells])
    ax.set_ylabel("macro-F1 drop")
    ax.set_ylim(0, 0.42)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Speaker-disjoint \u2192 source-disjoint drop", loc="left", pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig1_drops.{ext}")
    plt.close(fig)
    print("  wrote fig1_drops.pdf/.png")


def figure2():
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    for s in SEP:
        col = CORPUS_COLORS[s[0]]
        ax.scatter([s[2]], [s[3]], s=62, facecolor=col, edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.plot([s[4]], [s[3]], marker="|", color=col, markersize=9,
                alpha=0.55, zorder=2)

    offsets = {("AYDID", "programme"): (11, -3),
               ("SADA_S", "programme"): (11, -3),
               ("SADIC", "condition"): (11, -10),
               ("SADIC", "country"): (-12, 9)}
    for s in SEP:
        lab = s[0] if s[0] != "SADIC" else f"SADIC ({s[1]})"
        dx, dy = offsets[(s[0], s[1])]
        ax.annotate(lab, (s[2], s[3]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.8,
                    ha="right" if dx < 0 else "left",
                    color=CORPUS_COLORS[s[0]])

    ax.set_xlabel("within-label source separability (macro-F1)")
    ax.set_ylabel("speaker \u2192 source drop (macro-F1)")
    ax.set_xlim(0.0, 0.88)
    ax.set_ylim(-0.03, 0.40)
    ax.set_title("Source distinctiveness and evaluation inflation",
                 loc="left", pad=8)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#777777",
               markeredgecolor="black", markersize=6, label="measured"),
        Line2D([], [], marker="|", linestyle="", color="#777777",
               markersize=8, label="chance level"),
    ], frameon=False, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig2_separability.{ext}")
    plt.close(fig)
    print("  wrote fig2_separability.pdf/.png")


def figure3():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    protos = ["random", "speaker", "source"]
    xlabels = ["random", "speaker-\ndisjoint", "source-\ndisjoint"]
    markers = ["o", "s", "^"]

    for i, (corpus, vals) in enumerate(PROTO_ABS.items()):
        col = LINE_COLORS[corpus]
        ax.plot(range(3), [vals[p] for p in protos], marker=markers[i],
                color=col, linewidth=1.6, markersize=5.5,
                markeredgecolor="black", markeredgewidth=0.5, label=corpus)
        ax.axhline(CHANCE[corpus], color=col, linewidth=0.7,
                   linestyle=":", alpha=0.5)
        ax.text(2.07, CHANCE[corpus], "chance", fontsize=6.5, va="center",
                color=col, alpha=0.8)

    ax.set_xticks(range(3))
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.15, 2.5)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("macro-F1 (channel-only features)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Where the accuracy is lost", loc="left", pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig3_protocols.{ext}")
    plt.close(fig)
    print("  wrote fig3_protocols.pdf/.png")


def main():
    drops = dict(FALLBACK)
    drops.update(load_encoder_csvs())
    figure1(drops)
    figure2()
    figure3()
    print("""
Fig. 1. Loss in macro-F1 when whole recording sources are held out, relative to
speaker-disjoint evaluation, for three representations. Error bars are 95%
confidence intervals over twenty resamples where available; the SADIC condition
holdout admits two folds and is shown without intervals.

Fig. 2. Within-label source separability against the measured drop. Separability
is macro-F1 for predicting source from channel-only features within each label,
speaker-disjointly, so that label identity cannot contribute; tick marks give
each corpus's chance level.

Fig. 3. Absolute macro-F1 from channel-only features across the three protocols.
SADA's median of one clip per speaker unit makes its random-to-speaker segment
nearly flat by construction.
""")


if __name__ == "__main__":
    main()
