\# Recording-Source Bias in Arabic Dialect Identification



Code, manifests and fold definitions for the paper.



\## Setup



&#x20;   pip install -r requirements.txt



Set environment variables pointing at your corpora:



&#x20;   setx AYDID\_ROOT "D:\\path\\to\\aydid"

&#x20;   setx SADA\_ROOT  "D:\\path\\to\\sada"

&#x20;   setx ADI17\_ROOT "D:\\path\\to\\adi17"



\## Run



&#x20;   python -m analysis.protocols\_aydid

&#x20;   python -m analysis.protocols\_sada

&#x20;   python -m analysis.matched\_20\_draws

&#x20;   python -m analysis.source\_density

&#x20;   python -m analysis.vad\_validation

&#x20;   python -m analysis.v4\_reruns

&#x20;   python -m analysis.separability

&#x20;   python -m figures.make\_figures



\## Data



AYDID: <link>

SADA: <link>

ADI-17: HuggingFace ArabicSpeech/ADI17



Audio is not included. Manifests are in manifests/.

