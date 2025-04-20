import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import configparser
import os
import numpy as np
import librosa
import parselmouth
import re
import unicodedata
import inflect
from kanjize import number2kanji
from sudachipy import Dictionary, SplitMode
from functools import cache
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from .denoiser import Denoiser
from sklearn.model_selection import train_test_split

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

main_config = configparser.ConfigParser()
main_config.read('config/config.ini')
_punctuation = ';:,.!?¡¿—…"«»“”() *~-/\\&'


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


def _expand_decimal_point(m: re.Match) -> str:
    return m.group(1).replace(".", " point ")


def _expand_dollars(m: re.Match) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m: re.Match) -> str:
    return _inflect.number_to_words(m.group(0))

@cache
def get_backend(language: str) -> "EspeakBackend":
    import logging

    from phonemizer.backend import EspeakBackend

    logger = logging.getLogger("phonemizer")
    backend = EspeakBackend(
        language,
        preserve_punctuation=True,
        with_stress=True,
        punctuation_marks=_punctuation,
        logger=logger,
    )
    logger.setLevel(logging.ERROR)
    return backend

def _expand_number(m: re.Match) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    else:
        return _inflect.number_to_words(num, andword="")

class EmoDataset(Dataset):
    def __init__(self, target, reference, fold=1, language_code='en', randomize=True, device="cuda:0", denoise=True, seed=42, skip=0, audio_description=True, split='valid', balance=False, sample=False, force_balance=True, increase_max_count=1):
        self.target_data = pd.read_csv(os.path.join(main_config['DATA']['CSV_PATH'], f'{target}.csv'))
        train_labels = self.target_data[self.target_data['fold'] == fold][self.target_data['split'] == 'train']
        train_data, valid_data = train_test_split(train_labels, test_size=0.2, stratify=train_labels['emotion'], random_state=42)
        
        if split == 'train':
            self.target_data = train_data
        elif split == 'valid':
            self.target_data = valid_data
        
        self.target_data = self.target_data.reset_index(drop=True)
        
        if sample:
            self.target_data = self.target_data.groupby('emotion').apply(lambda x: x.sample(50, random_state=seed)).reset_index(drop=True)
        
        self.target_data = self.target_data.reset_index(drop=True)
        self.reference_data = pd.read_csv(os.path.join(main_config['DATA']['CSV_PATH'], f'{reference}.csv'))
        self.reference_data = self.reference_data[self.reference_data['fold'] == fold][self.reference_data['split'] == split]
        
        if self.reference_data.empty and split=="valid":
            self.reference_data = pd.read_csv(os.path.join(main_config['DATA']['CSV_PATH'], f'{reference}.csv'))
            train_labels = self.reference_data[self.reference_data['fold'] == fold][self.reference_data['split'] == 'train']
            _, self.reference_data = train_test_split(train_labels, test_size=0.2, stratify=train_labels['emotion'], random_state=42)
        
        self.reference_data = self.reference_data.reset_index(drop=True)
        
        # Balance
        if balance:
            ref_counts = self.reference_data['emotion'].value_counts()
            target_number = round(ref_counts.max() * increase_max_count)
            samples_to_add = []  # Lista para almacenar las muestras adicionales
            
            # Iterar sobre cada categoría presente en reference_data
            for emotion, count in ref_counts.items():
                # Calcular cuántas muestras se necesitan para llegar al objetivo
                needed = target_number - count
                if needed > 0:
                    # Filtrar target_data para la emoción
                    emotion_target_data = self.target_data[self.target_data['emotion'] == emotion]
                    available = len(emotion_target_data)
                    if available == 0:
                        continue

                    if needed < available:
                        sampled = emotion_target_data.sample(n=needed, random_state=42, replace=False)
                    elif force_balance:
                        sampled = emotion_target_data.sample(n=needed, random_state=42, replace=True)
                    else:
                        sampled = emotion_target_data.sample(n=available, random_state=42, replace=False)
                    samples_to_add.append(sampled)
            
            self.target_data = pd.concat(samples_to_add, ignore_index=True) if samples_to_add else pd.DataFrame(columns=self.reference_data.columns)

        else:
            self.target_data = self.target_data[self.target_data['emotion'].isin(self.reference_data['emotion'].unique())]

        self.audio_description = audio_description
        self.skip = skip
        self.resampler = None
        language_codes = {
            "en": "en-us",
            "es": "es",
            "pt": "pt-br",
            "eu": "eu",
            "ca": "ca",
        }
        self.effects = [
            ["silence", "1", "0.5", "1%", "-1", "0.5", "1%"]
        ]
        self.denoise = denoise
        self.denoiser = Denoiser().to(device) if denoise else None
        self.language_code = language_codes[language_code]
        if randomize:
            self.target_data = self.target_data.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.reference_data = self.reference_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        if target != reference and os.path.exists(os.path.join('out', 'evaluations', 'translations', f'{target}_{language_code}.csv')):
            self.translations = pd.read_csv(os.path.join('out', 'evaluations', 'translations', f'{target}_{language_code}.csv'))
        else:
            self.translations = None
            
        self.seed = seed

    def normalize_jp_text(self, text: str, tokenizer=Dictionary(dict="full").create()):
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
        final_text = " ".join([x.reading_form() for x in tokenizer.tokenize(text, SplitMode.A)])
        return final_text

    def normalize_numbers(self, text: str) -> str:
        text = re.sub(_comma_number_re, _remove_commas, text)
        text = re.sub(_pounds_re, r"\1 pounds", text)
        text = re.sub(_dollars_re, _expand_dollars, text)
        text = re.sub(_decimal_number_re, _expand_decimal_point, text)
        text = re.sub(_ordinal_re, _expand_ordinal, text)
        text = re.sub(_number_re, _expand_number, text)
        return text

    def clean(self, text, language):
        texts_out = []
        if not isinstance(text, str) and not isinstance(text, list):
            text = " "
        if "ja" in language:
            text = self.normalize_jp_text(text)
        else:
            text = self.normalize_numbers(text)
        texts_out.append(text)
        return texts_out

    def phonemize(self, text, language):
        text = self.clean(text, language)

        backend = get_backend(language)
        phonemes = backend.phonemize(text, strip=True, separator=Separator(word=" ", phone=","))

        if not phonemes:
            return 0
        else:
            return len(phonemes[0].split(','))

    def get_audio_features(self, path, trimmed_waveform, transcription, language_code):
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]
        pitch_std = np.std(pitch_values)
        y, sr = torchaudio.load(path)
        phonemes_len = self.phonemize(transcription, language_code)
        #non_silent_intervals = librosa.effects.split(y.cpu().numpy(), top_db=20)
        #duration = np.sum([(end - start) for start, end in non_silent_intervals]) / sr
        duration = trimmed_waveform.shape[1] / sr
        del y, sr, pitch_values
        speaking_rate = (phonemes_len / duration)

        return pitch_std, speaking_rate

    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        if idx < self.skip:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        emotion = self.target_data['emotion'][idx]
        target_audio_path = os.path.join(main_config['DATA']['FILES_PATH'],self.target_data['audio_path'][idx].replace('msppodcast', 'msppodcast/Audio'))
        reference_audio_path = self.reference_data[self.reference_data['emotion'] == emotion]['audio_path'].sample(1, random_state=idx+self.seed).values[0]
        
        if self.translations is not None:
            reference_transcription = self.reference_data[self.reference_data['audio_path'] == reference_audio_path]['transcription'].values[0]
            if not isinstance(reference_transcription, str):
                reference_transcription= " "
            reference_transcription = re.sub(r"[\[\]']", "", reference_transcription)
            reference_transcription = reference_transcription.replace("\n", "").replace("  ", " ").strip()
            target_transcription = self.translations[self.translations['audio_path'] == os.path.join(main_config['DATA']['FILES_PATH'],target_audio_path)]['translated_text'].values[0]
        else:
            target_transcription = self.target_data['transcription'][idx]
            if not isinstance(target_transcription, str):
                target_transcription= " "
            target_transcription = re.sub(r"[\[\]']", "", target_transcription)
            target_transcription = target_transcription.replace("\n", "").replace("  ", " ").strip()
            reference_transcription = target_transcription
        
        reference_audio_path = os.path.join(main_config['DATA']['FILES_PATH'], reference_audio_path)
        target_audio, target_sr = torchaudio.load(target_audio_path)
        if self.translations is not None:
            reference_audio, referece_sr = torchaudio.load(reference_audio_path)
        else:
            reference_audio, referece_sr = target_audio, target_sr
        
        if self.denoise:
            if referece_sr != 16000:
                if not self.resampler:
                    self.resampler = torchaudio.transforms.Resample(orig_freq=referece_sr, new_freq=16000)
                reference_audio = self.resampler(reference_audio)
                referece_sr = 16000
            reference_audio = self.denoiser(reference_audio, referece_sr, output_path=None, output_noise=False)

        trimmed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            reference_audio, referece_sr, self.effects
        )
        if trimmed_audio.shape[1] != 0:
            reference_audio = trimmed_audio

        if self.audio_description:
            pitch_std, speaking_rate = self.get_audio_features(target_audio_path, reference_audio, target_transcription, self.language_code)
        else:
            pitch_std, speaking_rate = 0, 0
    
        return target_audio_path, target_audio, target_transcription, target_sr, reference_audio_path, reference_audio, referece_sr, reference_transcription, pitch_std, speaking_rate, emotion
        
    