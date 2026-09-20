# Recording-Source Effects in Arabic Dialect Identification: A Cross-Corpus Evaluation

Code, manifests, and fold definitions for reproducing every value reported in
the paper.

Mansoor S. M. Ba Mahel,*, Wei Jianguoa, Xianghu Yuea, Norah Saeed Awnb, Abdulaziz S. Bamahelc
Submitted to *Speech Communication*.

---

## What this repository contains

The paper measures how much accuracy a dialect identification system loses when
whole recording sources are held out rather than whole speakers, and how that
loss differs across three Arabic corpora. It reports a second, independent
result: the low-energy-frame control widely used to argue that a signal is
non-phonetic is substantially contaminated with speech.

Audio is not redistributed. The manifests reference the original corpus
releases, and every script reads paths from environment variables.

---

## Setup

```
pip install -r requirements.txt
```

Point the scripts at your local copies of the corpora:

```
setx AYDID_ROOT "D:\path\to\aydid"
setx SADA_ROOT  "D:\path\to\sada"
setx ADI17_ROOT "D:\path\to\adi17"     # optional; ADI-17 streams from HuggingFace
```

ADI-17 is a gated HuggingFace dataset. Request access at
`huggingface.co/datasets/ArabicSpeech/ADI17`, then `hf auth login`. The
extraction script streams the parquet shards, so no local copy is required.

---

## Reproducing each table and figure

| Output | Script |
|---|---|
| Table 1 — corpus summary | `manifests/` (no computation) |
| Table 2 — absolute macro-F1, three protocols | `analysis.protocols_aydid`, `analysis.protocols_sada`, `analysis.adi17_protocols` |
| Table 3 — source-specific increment | `analysis.sada_rebalanced`, `analysis.protocols_aydid --increment` |
| Table 4, Figure 2 — matched comparison, 20 draws | `analysis.matched_20_draws` |
| Table 5, Figure 3 — source-density sensitivity | `analysis.source_density` |
| Table 6, Figure 4 — low-energy control validation | `analysis.vad_extended` |
| Table 7 — encoders with dimensionality matched | `analysis.encoder_vs_acoustic` |
| Table 8 — within-label source separability | `analysis.separability` |
| Section 3.2 — file-disjoint control | `analysis.file_disjoint_control` |
| Section 6.1 — unsupervised separability (null) | `analysis.unsupervised_separability` |
| Figure 1 — protocol ladder | `figures.make_figures` |

Feature extraction must run first:

```
python -m features.extract_acoustic --corpus aydid
python -m features.extract_acoustic --corpus sada
python -m features.extract_adi17
python -m features.extract_encoder --corpus aydid --encoder wavlm
python -m features.extract_encoder --corpus aydid --encoder xlsr
python -m features.extract_encoder --corpus sada  --encoder wavlm
python -m features.extract_encoder --corpus sada  --encoder xlsr
```

Encoder extraction needs a GPU; everything else runs on CPU.

---

## Headline values

For checking a reproduction against the paper.

| Quantity | AYDID | SADA_S | ADI-17 |
|---|---|---|---|
| Random → source (acoustic) | +0.065 | +0.289 | +0.243 |
| Speaker → source increment | +0.018 | +0.301 † | n/a ‡ |
| Matched, 20 draws | +0.022 ± 0.082 | +0.291 ± 0.033 | n/a ‡ |
| Within-label source separability | 0.183 (1.03× chance) | 0.516 (11.04× chance) | n/a |
| Low-energy frames inside speech (Silero) | 0.615 | 0.623 | 0.707 |

† On a restricted, class-balanced subset of 1,224 of 7,499 clips; see
Section 3.2. ‡ ADI-17 provides no speaker labels.

---

## Fold definitions

All folds are released under `folds/` and can be regenerated with
`folds/generate_folds.py`.

Tables 2 and 3 share one fold set. **Table 7 uses an independent draw of the
same construction**, which is why its acoustic baseline differs: 0.560 against
0.576 on AYDID, and 0.297 against 0.234 on SADA_S. Both differences lie within
the fold-to-fold variability reported in Table 2 (0.041 and 0.131
respectively). See Section 5.3.

Resamples are grouped draws taken with replacement from the space of held-out
sets, and share training data extensively. They are not independent
replications. The intervals in the paper indicate spread, not inferential
uncertainty, and no p-values are reported.

---

## Notes on two analyses

**The matched comparison (Table 4).** AYDID contributes roughly fifty clips per
speaker and SADA_S roughly five, so the two arms cannot be matched on total
clips and on clips-per-speaker simultaneously. We match total clips; the
residual size difference is about 12%. Both arms are averaged over twenty
independent draws because both involve an arbitrary selection — which three of
seven AYDID dialects, and which SADA_S speaker units survive rebalancing. An
earlier six-draw version gave disjoint ranges; that separation does not survive
at twenty draws, and the released code reproduces the larger sample.

**The low-energy control (Table 6).** Two detectors are run: WebRTC, which
thresholds on energy, and Silero, which does not. Each is also applied to the
high-energy frames of the same utterances as a positive control. WebRTC's
control fails on SADA_S (0.537) and ADI-17 (0.314), so its low-energy figures
for those corpora are not interpretable and no inference is drawn from them.
The reported result rests on Silero, whose control passes everywhere
(0.920–0.952).

---

## Data

- **AYDID** — released with program annotations: `<link>`
- **SADA** — used under its existing public release: `<link>`
- **ADI-17** — gated HuggingFace dataset `ArabicSpeech/ADI17` (Shon et al., 2020)
- **SADIC** — not analyzed in this paper; a corpus-quality finding is reported
  in the supplementary material

---

## Layout

```
manifests/    per-clip manifests for all three corpora
folds/        fold assignments, plus generate_folds.py
features/     extraction scripts (acoustic, encoder, ADI-17 streaming)
analysis/     one script per table
figures/      make_figures.py
results/      the exact values printed in the manuscript
```

---

## Citation

```
@article{bamahel2026recording,
  title   = {Recording-Source Effects in Arabic Dialect Identification:
             A Cross-Corpus Evaluation},
  author  = {Ba Mahel, Mansoor S. M. and Wei, Jianguo and Yue, Xianghu
             and Awn, Norah Saeed and Bamahel, Abdulaziz S.},
  journal = {Speech Communication},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code MIT. Manifests and derived features CC-BY-4.0. Audio is not redistributed
and remains under the terms of the original corpus releases.
