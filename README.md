# Latin ITN (`latin-itn`)

Inverse Text Normalization (Casing & Trailing Punctuation Restoration) for Latin ASR transcriptions.

## Installation

```bash
pip install latin-itn

```

## Quickstart

```python
from latin_itn import LatinITNPredictor

predictor = LatinITNPredictor.from_pretrained("njand/latin-asr-postprocessor")

raw_text = "gallia est omnis divisa in partes tres quarum unam incolunt belgae"
formatted_text = predictor.predict(raw_text)

print(formatted_text)
# Output: "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae."

```
