# SpeechFactory

*A framework for synthesising multilingual emotional speech data for Speech Emotion Recognition (SER) research.*

---

## About
SpeechFactory provides a reproducible pipeline to generate **synthetic, acoustically‑rich emotional speech corpora** using Text‑to‑Speech (TTS) and automatic translation.  
The current release focuses on the main languages of the Iberian Peninsula (Spanish, Catalan, Galician, Basque and Portuguese) but can be extended to any language supported by your chosen TTS system.

> **Why?** Public SER datasets are limited in size, language coverage and emotion balance.  Synthetic data helps bridge this gap and improves cross‑lingual SER generalisation, as shown in our accompanying paper:
>
> > J. Bellver *et al.* “**Multilingual Speech Emotion Recognition in Iberian Languages: A Generative AI Framework with LLMs and TTS Data Augmentation**”.

Please cite the paper if you use this repository in your work.

---

## Repository layout

```
SpeechFactory/
├── config/
│   └── config.ini          # Paths to raw & processed data
├── requirements.txt        # Python dependencies
└── src/
    ├── …                   # Core implementation
    └── pipeline/
        ├── translations_pipeline.py  # ↳ Step 1: text translation
        └── tts_pipeline.py           # ↳ Step 2: TTS synthesis
```

### Key folders
| Folder / file | Purpose |
|---------------|---------|
| **`config/config.ini`** | Two required keys:<br>`CSV_PATH` → root folder that contains *sub‑folders* for each *raw* database (`*.wav`).<br>`FILES_PATH` → root folder that contains the *processed* Emobox‑formatted JSON/JSON‑CLI data. |
| **`src/`** | All Python code.  Important sub‑packages: `asr`, `denoiser`, `translator`, `tts`, `embedders`, etc. |
| **`src/pipeline/`** | Two high‑level scripts that orchestrate the full data‑generation workflow. |

---

## Quick start

### 1. Clone & install
```bash
# Clone your fork (or the upstream repo)
git clone https://github.com/your‑username/SpeechFactory.git
cd SpeechFactory

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Download required model weights
Place all checkpoints inside **`models/`** (create if it does not exist):

* **ERM emotion classifiers** (see paper or release page for links)
* **Physics‑Pitch TTS** weights

Update paths in any YAML/JSON the models need, if applicable.

### 3. Configure data locations
Edit **`config/config.ini`** so that:

```ini
[DATA]
CSV_PATH   = /absolute/path/to/raw_databases
FILES_PATH = /absolute/path/to/processed_databases
```
`CSV_PATH` must contain *one sub‑directory per source database*, each with the corresponding `.wav` files.  
`FILES_PATH` must mirror the EmoBox JSON/JSON‑CLI structure for each processed database.

---

## Running the pipeline

> **Tip:** All scripts accept `-h / --help`.

### Step 1 – Translation
```bash
python src/pipeline/translations_pipeline.py \
    --reference_database   "MEACorpus" \         # database to translate FROM
    --reference_language "es" \            # ISO‑639‑1 of destination language (e.g. es, pt, eu, gl)
    --target_database     "MSPPodcast" \
    --target_language "en"
```
This script creates translated transcriptions for each utterance and stores them under the processed path.

### Step 2 – TTS synthesis
```bash
python src/pipeline/tts_pipeline.py \
    --reference_database   "MSPPodcast" \        
    --reference_language "en" \ 
    --target_database     "MEACorpus" \
    --target_language "es"
```
TTS voices each translated utterance with the desired emotional style, creating a parallel multilingual corpus.


---


## Contact
For questions, open an issue or contact: jaime.bellver@upm.es

