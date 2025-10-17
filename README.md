# 🏭 SpeechFactory

*A framework for synthesising multilingual emotional speech data for Speech Emotion Recognition (SER) research.*

---

## 📖 About
SpeechFactory provides a reproducible pipeline to generate **synthetic, acoustically-rich emotional speech corpora** using Text-to-Speech (TTS) and automatic translation.  
The current release focuses on the main languages of the Iberian Peninsula (Spanish, Catalan, Galician, Basque and Portuguese) but can be extended to any language supported by your chosen TTS system.

> **Why?** Public SER datasets are limited in size, language coverage and emotion balance. Synthetic data helps bridge this gap and improves cross-lingual SER generalisation, as shown in our accompanying paper:
>
> > J. Bellver-Soler *et al.* "**Multilingual Speech Emotion Recognition in Iberian Languages: A Generative AI Framework with LLMs and TTS Data Augmentation**".

If you use this repository in your work, please cite the following paper:

**BibTeX:**
```bibtex
@inproceedings{bellver2025multilingual,
  title     = {Multilingual Speech Emotion Recognition in Iberian Languages: A Generative AI Framework with LLMs and TTS Data Augmentation},
  author    = {Bellver-Soler, Jaime et al.},
  year      = {2025},
  booktitle = {},
  publisher = {},
}
```

---

## 🔊 Example Generated Dataset

You can explore a sample dataset generated with SpeechFactory on HuggingFace🤗:

SER-MSPMEA-Spanish Database: Combines MSPPodcast with MEACorpus. Download the spanish synthetic audios [here](https://huggingface.co/datasets/jaimebellver/SER-MSPMEA-Spanish).

## ✨ SER Model Weights

The custom Speech-Emotion-Recognition (SER) model developed and used in our paper is available on HuggingFace🤗. You can download the pretrained weights and configuration files [here](https://huggingface.co/jaimebellver/whisper-large-v3-SER).

## 🗂️ Repository layout

```
SpeechFactory/
├── config/
│   └── config.ini              # Paths to raw & processed data
├── requirements.txt            # Python dependencies
├── model_weights/              # (You must create this) Checkpoints go here
├── src/
│   ├── asr/                    # Optional ASR utilities
│   ├── denoiser/               # Optional audio denoising
│   ├── embedders/              # Embedding utils for emotion features
│   ├── evaluator/
│   │   └── WIFER-HACER/         # Speech-to-Motion Recognition model files used in the paper; replaceable
│   ├── translator/             # Translation logic
│   ├── tts/                    # TTS synthesis engines
│   ├── generator/
│   │   ├── phish-speech/        # Cloned FishSpeech repo
│   │   └── fish_speech_wrapper.py  # Wrapper to interact with FishSpeech
│   └── pipeline/
│       ├── translations_pipeline.py   # → Step 1: text translation
│       └── tts_pipeline.py           # → Step 2: TTS synthesis
```  

### Key folders
| Folder / file | Purpose |
|---------------|---------|
| **`config/config.ini`** | Two required keys:<br>`CSV_PATH` → root folder containing CSVs for each raw database.<br>`FILES_PATH` → root folder with processed Emobox JSON/JSON-CLI data. |
| **`src/`** | Core Python code. Important sub-packages: `asr`, `denoiser`, `translator`, `tts`, `embedders`, `generator` (contains phish-speech). |
| **`src/pipeline/`** | Orchestrates the full pipeline: translation & TTS. |
| **`src/generator/phish-speech/`** | Clone of [FishSpeech](https://github.com/fishaudio/fish-speech). Used via wrapper `fish_speech_wrapper.py`. Not optimised, but functional. |

## 📊 Data setup

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
| `transcription`| Text transcription of the audio |
| `fold`         | (Optional) Numeric fold ID, if dataset includes folds. Omit or use 1 if not applicable. |
| `split`        | Data split: `train`, `valid`, or `test` |

---

## 🏃‍♀️ Quick start

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

## 🛠️ Running the pipeline

> **Tip:** All scripts accept `-h / --help` for argument details.

### Step 1 – Translation
```bash
python src/pipeline/translations_pipeline.py \
    --reference_database   "msppodcast" \
    --reference_language   "en" \
    --target_database      "meacorpus" \
    --target_language      "es"
```
Creates translated transcriptions and stores them under the processed path (`FILES_PATH`).

### Step 2 – TTS synthesis
```bash
python src/pipeline/tts_pipeline.py \
    --reference_database   "meacorpus" \
    --reference_language   "es" \
    --target_database      "msppodcast" \
    --target_language      "en"
```
Voices the translated utterances using the selected emotional TTS voice. Saves output in parallel multilingual format.

---

## 📋 Contact
For questions, open an issue or contact: jaime.bellver@upm.es

---

## 📜 License

Released under **CC BY 4.0**. Attribution required for derivative works.
Note: MEACorpus includes YouTube-sourced content — additional rights may apply.

---

## 🙌 Acknowledgements

Supported by:

* **European Commission** – ASTOUND3 (101071191, Horizon Europe)
* **MCIN/AEI/ERDF** – Project BEWORD (PID2021-126061OB-C43)
* **INNOVATRAD-CM** – Comunidad de Madrid (PHS-2024/PH-HUM-52)

---

**Authors:** Jaime Bellver-Soler, Anmol Guragain, Samuel Ramos-Varela, Ricardo Córdoba, Luis Fernando D’Haro
*Speech Technology and Machine Learning Group, Universidad Politécnica de Madrid*

