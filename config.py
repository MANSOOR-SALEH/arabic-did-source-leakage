"""
Central configuration for arabic-did-source-leakage.

No absolute paths. Every dataset root is read from an environment variable.

Set these before running any script:

    export AYDID_ROOT=/path/to/aydid          (Linux/macOS)
    export SADA_ROOT=/path/to/sada
    export ADI17_ROOT=/path/to/adi17

    setx AYDID_ROOT "F:\DID_ASR_Datasets\AYDID"   (Windows, once)
    setx SADA_ROOT  "F:\DID_ASR_Datasets\SADA_S"
    setx ADI17_ROOT "F:\DID_ASR_Datasets\ADI17"
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
MANIFESTS = REPO_ROOT / "manifests"
FOLDS = REPO_ROOT / "folds"
FEATURES = REPO_ROOT / "features"
FEATURES_CACHE = REPO_ROOT / "features" / "cache"
RESULTS = REPO_ROOT / "results"
LOGS = RESULTS / "logs"
FIGURES = REPO_ROOT / "figures"

for _d in (MANIFESTS, FOLDS, FEATURES, FEATURES_CACHE, RESULTS, LOGS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset roots (environment variables, no absolutes)
# ---------------------------------------------------------------------------

AYDID_ROOT = Path(os.environ.get("AYDID_ROOT", ""))
SADA_ROOT  = Path(os.environ.get("SADA_ROOT", ""))
ADI17_ROOT = Path(os.environ.get("ADI17_ROOT", ""))

# ---------------------------------------------------------------------------
# Manifests (released in this repository)
# ---------------------------------------------------------------------------

AYDID_MANIFEST       = MANIFESTS / "aydid_manifest.csv"
SADA_S_MANIFEST      = MANIFESTS / "sada_s_manifest.csv"
SADA_S_RESTRICTED    = MANIFESTS / "sada_s_restricted.csv"
ADI17_MANIFEST       = MANIFESTS / "adi17_manifest.csv"
SADA_S_REBALANCED_DIR = MANIFESTS / "sada_s_rebalanced"

# Local working copies inside each dataset root (used by the extractors)
AYDID_SOURCE_MANIFEST = AYDID_ROOT / "aydid_master.csv"
SADA_SOURCE_MANIFEST  = SADA_ROOT / "sada_clean.csv"

# Audio roots
AYDID_WAV_ROOTS = [AYDID_ROOT / "WAVS_16K", AYDID_ROOT / "WAVS"]
SADA_WAV_ROOTS  = [SADA_ROOT, SADA_ROOT / "WAVS"]

# ---------------------------------------------------------------------------
# Feature caches (regenerable; not committed)
# ---------------------------------------------------------------------------

FEAT_AYDID_FULL       = FEATURES_CACHE / "aydid_channel_features_full.csv"
FEAT_AYDID_SPEECH     = FEATURES_CACHE / "aydid_channel_features_speech.csv"
FEAT_AYDID_NONSPEECH  = FEATURES_CACHE / "aydid_channel_features_nonspeech.csv"
FEAT_AYDID_CMVN       = FEATURES_CACHE / "aydid_channel_features_cmvn.csv"

FEAT_SADA_FULL        = FEATURES_CACHE / "sada_channel_features_full.csv"
FEAT_SADA_SPEECH      = FEATURES_CACHE / "sada_channel_features_speech.csv"
FEAT_SADA_NONSPEECH   = FEATURES_CACHE / "sada_channel_features_nonspeech.csv"
FEAT_SADA_CMVN        = FEATURES_CACHE / "sada_channel_features_cmvn.csv"

FEAT_ADI17_FULL       = FEATURES_CACHE / "adi17_channel_features_full.csv"
FEAT_ADI17_NONSPEECH  = FEATURES_CACHE / "adi17_channel_features_nonspeech.csv"

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

CORPORA = {
    "aydid": {
        "root": AYDID_ROOT,
        "manifest": AYDID_MANIFEST,
        "source_manifest": AYDID_SOURCE_MANIFEST,
        "wav_roots": AYDID_WAV_ROOTS,
        "n_classes": 7,
        "chance": 1 / 7,
        "grouping": "show_name",
    },
    "sada": {
        "root": SADA_ROOT,
        "manifest": SADA_S_MANIFEST,
        "source_manifest": SADA_SOURCE_MANIFEST,
        "wav_roots": SADA_WAV_ROOTS,
        "n_classes": 3,
        "chance": 1 / 3,
        "grouping": "show_name",
    },
    "adi17": {
        "root": ADI17_ROOT,
        "manifest": ADI17_MANIFEST,
        "n_classes": 17,
        "chance": 1 / 17,
        "grouping": "video",
    },
}

# ---------------------------------------------------------------------------
# Audio / feature extraction (Section 4.2)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
MONO = True
STFT_WINDOW = 512
STFT_HOP = 160
N_MELS = 26
N_MFCC = 20

NOISE_FLOOR_PERCENTILE = 10
SNR_HIGH_PERCENTILE = 90
LOW_ENERGY_PERCENTILE = 30
HIGH_ENERGY_PERCENTILE = 70

# ---------------------------------------------------------------------------
# Encoders (Section 4.3)
# ---------------------------------------------------------------------------

ENCODERS = {
    "wavlm": {
        "name": "microsoft/wavlm-large",
        "layers": ["one_third", "two_thirds", "final"],
        "pooling": "mean+std",
        "dim": 2048,
    },
    "xlsr": {
        "name": "facebook/wav2vec2-xls-r-300m",
        "layers": ["one_third", "two_thirds", "final"],
        "pooling": "mean+std",
        "dim": 2048,
    },
}
MAX_CLIP_SECONDS = 20.0

# ---------------------------------------------------------------------------
# Classifier (Section 4.3)
# ---------------------------------------------------------------------------

LR_KWARGS = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)

# ---------------------------------------------------------------------------
# Evaluation (Section 4.1)
# ---------------------------------------------------------------------------

N_RESAMPLES = 20
N_REBALANCED_SUBSAMPLES = 5
MACRO_F1_CONVENTION = "mean_of_fold"