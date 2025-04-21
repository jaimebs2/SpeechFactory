# SpeechFactory

*A framework for synthesising multilingual emotional speech data for Speech Emotion Recognition (SER) research.*

---

## About
SpeechFactory provides a reproducible pipeline to generate **synthetic, acoustically-rich emotional speech corpora** using Text-to-Speech (TTS) and automatic translation.  
The current release focuses on the main languages of the Iberian Peninsula (Spanish, Catalan, Galician, Basque and Portuguese) but can be extended to any language supported by your chosen TTS system.

> **Why?** Public SER datasets are limited in size, language coverage and emotion balance. Synthetic data helps bridge this gap and improves cross-lingual SER generalisation, as shown in our accompanying paper:
>
> > J. Bellver *et al.* "**Multilingual Speech Emotion Recognition in Iberian Languages: A Generative AI Framework with LLMs and TTS Data Augmentation**".

If you use this repository in your work, please cite the following paper:

**BibTeX:**
```bibtex
@inproceedings{bellver2024multilingual,
  title     = {Multilingual Speech Emotion Recognition in Iberian Languages: A Generative AI Framework with LLMs and TTS Data Augmentation},
  author    = {Bellver, Jaime and others},
  year      = {2024},
  booktitle = {Proceedings of the [relevant conference]},
  publisher = {[Publisher]},
}
```

---

## Repository layout

```
SpeechFactory/
├── config/
│   └── config.ini              # Paths to raw & processed data
├── requirements.txt              # Python dependencies
├── model_weights/               # (You must create this) Checkpoints go here
├── src/
│   ├── asr/                    # Optional ASR utilities
│   ├── denoiser/               # Optional audio denoising
│   ├── embedders/             # Embedding utils for emotion features
│   ├── translator/            # Translation logic
│   ├── tts/                   # TTS synthesis engines
│   └── pipeline/
│       ├── translations_pipeline.py   # → Step 1: text translation
│       └── tts_pipeline.py           # → Step 2: TTS synthesis
└── FishSpeech/               # FishSpeech repo clone; used via wrapper
```

### Key folders
| Folder / file | Purpose |
|---------------|---------|
| **`config/config.ini`** | Two required keys:<br>`CSV_PATH` → root folder containing CSVs for each raw database.<br>`FILES_PATH` → root folder with processed Emobox JSON/JSON-CLI data. |
| **`src/`** | Core Python code. Important sub-packages: `asr`, `denoiser`, `translator`, `tts`, `embedders`, etc. |
| **`src/pipeline/`** | Orchestrates the full pipeline: translation & TTS. |
| **`FishSpeech/`** | Clone of [FishSpeech](https://github.com/yzhou359/FishSpeech). Used via FishSpeechWrapper. Not optimised, but functional. |

---

## Data setup

Configure **`config/config.ini`**:

```ini
[DATA]
CSV_PATH   = /absolute/path/to/raw_csvs
FILES_PATH = /absolute/path/to/processed_databases
```

The `CSV_PATH` directory must contain a `.csv` file per source database, named accordingly (e.g. `meacorpus.csv`, `mspodcast.csv`, etc).

Each CSV must include the following **columns**:

| Column         | Description |
|----------------|-------------|
| `audio_path`   | Relative path to the `.wav` file. Use format: `<database_name>/<filename>.wav` |
| `emotion`      | Emotion label (e.g., `angry`, `sad`, `neutral`, etc.) |
| `Transcription`| Text transcription of the audio |
| `fold`         | (Optional) Numeric fold ID, if dataset includes folds. Omit or use 1 if not applicable. |
| `split`        | Data split: `train`, `valid`, or `test` |

---

## Quick start

### 1. Clone & install
```bash
# Clone your fork (or the upstream repo)
git clone https://github.com/your-username/SpeechFactory.git
cd SpeechFactory

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download model weights
Place required checkpoints in the **`model_weights/`** folder (create if it doesn't exist).

### 3. Prepare data files
Ensure the structure under `CSV_PATH` and `FILES_PATH` follows the format described above.

---

## Running the pipeline

> **Tip:** All scripts accept `-h / --help` for argument details.

### Step 1 – Translation
```bash
python src/pipeline/translations_pipeline.py \
    --reference_database   "MEACorpus" \
    --reference_language   "es" \
    --target_database      "MSPPodcast" \
    --target_language      "en"
```
Creates translated transcriptions and stores them under the processed path (`FILES_PATH`).

### Step 2 – TTS synthesis
```bash
python src/pipeline/tts_pipeline.py \
    --reference_database   "MSPPodcast" \
    --reference_language   "en" \
    --target_database      "MEACorpus" \
    --target_language      "es"
```
Voices the translated utterances using the selected emotional TTS voice. Saves output in parallel multilingual format.

---

## Contact
For questions, open an issue or contact: jaime.bellver@upm.es

