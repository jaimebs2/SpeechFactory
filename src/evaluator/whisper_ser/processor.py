from transformers import AutoProcessor
import librosa
from torch import nn
import torch

class AudioPreprocessor(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def preprocess(self, audio, sr):
        if sr != self.processor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.processor.sampling_rate)
        inputs = self.processor(
            raw_speech=audio,
            return_tensors='pt',
            sampling_rate=self.processor.sampling_rate,
            return_attention_mask=True
        )
        if 'input_features' not in inputs:
            inputs['input_features'] = inputs[list(inputs.keys())[0]]
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = inputs[list(inputs.keys())[-1]]
        return {
            'input_features': inputs['input_features'],
            'feature_attention_mask': inputs['attention_mask']
        }

class WhisperPreprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3").feature_extractor
        self.audio_preprocessor = AudioPreprocessor(self.audio_processor)

    def __call__(self, audio, sr):
        device = audio.device
        audio_inputs_processed = self.audio_preprocessor.preprocess(audio.cpu().numpy(), sr)
        inputs = {
            'input_features': audio_inputs_processed['input_features'].to(device, torch.bfloat16),
            'feature_attention_mask': audio_inputs_processed['feature_attention_mask'].to(device)
        }
        return inputs
