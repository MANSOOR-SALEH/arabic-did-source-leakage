\# Features



Acoustic: 61 statistics per utterance.

\- 26 mel band means

\- spectral centroid, rolloff, bandwidth, flatness (mean + std)

\- spectral tilt, noise floor, SNR

\- zero-crossing rate (mean + std)

\- RMS (mean + std)

\- 20 MFCC means



Variants: full, speech (top 30% energy), nonspeech (bottom 30%), cmvn.



Encoder: WavLM-Large, XLS-R-300M. Layers 1/3, 2/3, final averaged.

Mean+std pooling. 2048 dims.



Run:

&#x20;   python -m features.extract\_acoustic

&#x20;   python -m features.extract\_acoustic\_sada

&#x20;   python -m features.extract\_encoder --corpus aydid --encoder wavlm

