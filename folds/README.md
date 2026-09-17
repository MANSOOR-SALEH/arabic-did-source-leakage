# Fold definitions

Three AYDID fold files are released:

- `table2_random.csv`   — fold → row index into the AYDID manifest
- `table2_speaker.csv`  — fold → held-out speaker id (within show)
- `table2_source.csv`   — fold → held-out show name

These were produced by `aydid_prepare.py` and
`aydid_folds_speaker_within_show.py` in the original project and are copied
here unchanged.

All other folds are generated inside their analysis scripts at runtime from
fixed seeds, and are not persisted:

| Fold group | Script |
|---|---|
| SADA protocols (Table 2/3) | `analysis/protocols_sada.py` |
| ADI-17 source folds (Table 2) | `analysis/adi17_protocols.py` |
| Table 4 matched draws | `analysis/matched_20_draws.py` |
| Table 5 density draws | `analysis/source_density.py` |
| Table 7 encoder folds | `analysis/encoder_vs_acoustic.py`, `analysis/v4_reruns.py` (R1) |
| Table 8 separability folds | `analysis/separability.py` |

A unified `generate_folds.py` is not released because the seed logic and the
fold-construction code are corpus-specific and would have to be reimplemented
in a single file, risking divergence from the reported results. The seeds used
are visible in each script.