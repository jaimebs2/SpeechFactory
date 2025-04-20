import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa  # Needed for Canary1bProcessor
from typing import Tuple, Optional
from transformers import HubertForCTC, Wav2Vec2ForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2Processor
from nemo.collections.asr.models import EncDecMultiTaskModel
from speechbrain.inference.speaker import EncoderClassifier

# ---------------------------------------------------------------------
# Audio loading & helper functions
# ---------------------------------------------------------------------

def load_and_preprocess_audio(file_path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load an audio file, resample it if needed, convert to mono,
    and create an attention mask.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    attention_mask = torch.ones(1, waveform.size(1), dtype=torch.long)
    return waveform, attention_mask

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding."""
    assert channels % 2 == 0
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).reshape(-1, 1) * inv_timescales.reshape(1, -1)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def whisper_attention_mask(x, feature_attention_mask):
    """Create attention mask for the Whisper model."""
    batch_size = x.shape[0]
    sequence_length = x.shape[2] // 2
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=x.device)
    audio_output_lengths = torch.tensor([sequence_length] * batch_size, device=x.device)
    
    return attention_mask, audio_output_lengths

# -----------------------------------------------------------------------------
# Base Embedder classes
# -----------------------------------------------------------------------------

class BaseEmbedder(nn.Module):
    """
    Base class for audio embedders that follow the pattern:
    - Run the encoder to get raw hidden states
    - Pool (average) over the time dimension to obtain a fixed-size representation
    """
    def forward(self, x, attention_mask):
        embeddings = self.encode(x, attention_mask)
        # If encode returns a tuple, assume the first element is the hidden states.
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        # If the result is time-varying (3D tensor), average over time.
        if embeddings.dim() == 3:
            pooled = torch.mean(embeddings, dim=1)
            return pooled
        return embeddings

    def encode(self, x, attention_mask):
        raise NotImplementedError("Each embedder must implement its own encode method.")

# -----------------------------------------------------------------------------
# Embedder Implementations
# -----------------------------------------------------------------------------

class HubertEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").hubert

    def encode(self, x, attention_mask):
        # Ensure correct input shape.
        if x.dim() == 3:
            x = x.squeeze(1)
        hidden_states = self.model(x)
        return hidden_states['last_hidden_state']

class CanaryEmbedder(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        self.decode_cfg = self.model.cfg.decoding
        self.decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(self.decode_cfg)

    def encode(self, x, attention_mask):
        self.model.encoder.to(dtype=torch.float32)
        encoded, _ = self.model.encoder(
            audio_signal=x.to(torch.float32), 
            length=attention_mask.sum(dim=1).to(torch.float32)
        )
        hidden_states = encoded.permute(0, 2, 1).to(x.device, x.dtype)
        return hidden_states

class WhisperEmbedder(BaseEmbedder):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        if checkpoint_path:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3", use_safetensors=True
            ).model.encoder
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3", use_safetensors=True
            ).model.encoder
        torch.nn.init.constant_(self.model.layer_norm.bias, 0.0)
        torch.nn.init.constant_(self.model.layer_norm.weight, 1.0)
        self.register_buffer("positional_embedding", sinusoids(1500, 1280))

    def encode(self, x, attention_mask):
        # Create an attention mask specific for Whisper.
        attn_mask, _ = whisper_attention_mask(x, attention_mask)
        x = nn.functional.gelu(self.model.conv1(x))
        x = nn.functional.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        x = x + self.model.embed_positions.weight
        for block in self.model.layers:
            x, _ = block(x, attn_mask, layer_head_mask=None, output_attentions=True)
        return x

class Wav2Vec2Embedder(BaseEmbedder):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").wav2vec2

    def encode(self, x, attention_mask):
        output = self.model(x, attention_mask)
        return output['last_hidden_state']

# For speaker models (which already produce fixed-size embeddings),
# we override forward instead of using the base pooling.
class EcapaEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        
    def forward(self, x, attention_mask=None):
        if x.dim() == 3:
            x = x.squeeze(1)
        embeddings = self.model.encode_batch(x)
        return embeddings.squeeze(1)

class XVectorEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
        
    def forward(self, x, attention_mask=None):
        if x.dim() == 3:
            x = x.squeeze(1)
        embeddings = self.model.encode_batch(x)
        return embeddings.squeeze(1)

# -----------------------------------------------------------------------------
# Global Cosine Similarity Class (modified to use processors)
# -----------------------------------------------------------------------------

class GlobalCosineSimilarity:
    """
    Given a model type (e.g., "hubert", "whisper", "canary", etc.),
    this class instantiates the corresponding embedder and—if specified—the
    associated processor. When a processor is used, the input audio file(s)
    are first preprocessed to obtain the necessary features and masks.
    """
    def __init__(self, model_type: str, device: str = "cuda", use_processor: bool = False, **kwargs):
        model_type = model_type.lower()
        self.use_processor = use_processor

        if model_type == "hubert":
            self.embedder = HubertEmbedder().to(device)
            self.processor = HubertPreprocess() if self.use_processor else None

        elif model_type == "canary":
            self.embedder = CanaryEmbedder().to(device)
            self.processor = Canary1bProcessor().to(device) if self.use_processor else None

        elif model_type == "whisper":
            self.embedder = WhisperEmbedder(checkpoint_path=kwargs.get("checkpoint_path")).to(device)
            self.processor = WhisperPreprocess() if self.use_processor else None

        elif model_type == "wav2vec2":
            self.embedder = Wav2Vec2Embedder().to(device)
            self.processor = Wav2Vec2Large960hProcessor() if self.use_processor else None

        elif model_type == "ecapa":
            self.embedder = EcapaEmbedder().to(device)
            self.processor = None

        elif model_type == "xvector":
            self.embedder = XVectorEmbedder().to(device)
            self.processor = None

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.embedder.eval()

    def embed_audio(self, audio_path: str) -> torch.Tensor:
        device = next(self.embedder.parameters()).device
        if self.processor is not None:
            # The processor expects a list of audio paths.
            inputs = self.processor([audio_path])
            input_features = inputs['input_features'].to(device)
            attention_mask = inputs['feature_attention_mask'].to(device)
            with torch.no_grad():
                embeddings = self.embedder(input_features, attention_mask)
        else:
            audio, attention_mask = load_and_preprocess_audio(audio_path)
            audio = audio.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.no_grad():
                embeddings = self.embedder(audio, attention_mask)
        return embeddings

    def __call__(self, audio_path1: str, audio_path2: str) -> float:
        emb1 = self.embed_audio(audio_path1)
        emb2 = self.embed_audio(audio_path2)
        # Compute cosine similarity (assumes embeddings are 2D: [batch, features])
        similarity = F.cosine_similarity(emb1, emb2, dim=1).item()
        return similarity

# -----------------------------------------------------------------------------
# Processors
# -----------------------------------------------------------------------------

class AudioPreprocessor(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def preprocess(self, audio_paths):
        audios = []
        for audio_path in audio_paths:
            audio, _ = librosa.load(
                audio_path,
                sr=self.processor.sampling_rate
            )
            audios.append(audio)
        inputs = self.processor(
            raw_speech=audios[0],
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


class HubertPreprocess(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        # Make sure sampling rate is set to 16000.
        self.audio_processor.__setattr__("sampling_rate", 16000)
        self.audio_preprocessor = AudioPreprocessor(self.audio_processor)

    def __call__(self, audio_paths):
        audio_inputs_processed = self.audio_preprocessor.preprocess(audio_paths)
        inputs = {
            'input_features': audio_inputs_processed['input_features'],
            'feature_attention_mask': audio_inputs_processed['feature_attention_mask']
        }
        return inputs

class Wav2Vec2ProcessorBase(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import Wav2Vec2Processor as W2VP
        self.processor = W2VP.from_pretrained(model_name)
        self.audio_preprocessor = AudioPreprocessor(self.processor)
        self.processor.sampling_rate = self.processor.feature_extractor.sampling_rate
        
    def to(self, device):
        self.device = device
        return self

    def __call__(self, audio_paths):
        inputs = self.audio_preprocessor.preprocess(audio_paths)
        return inputs
    
class Wav2Vec2LargeProcessor(Wav2Vec2ProcessorBase):
    def __init__(self, **kwargs):
        super().__init__("facebook/wav2vec2-large")

class Wav2Vec2Large960hProcessor(Wav2Vec2ProcessorBase):
    def __init__(self, **kwargs):
        super().__init__("facebook/wav2vec2-base-960h")

class Canary1bProcessor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').preprocessor
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __call__(self, audio_paths):
        audios = []
        for audio_path in audio_paths:
            audio, _ = librosa.load(
                audio_path,
                sr=self.model._sample_rate
            )
            audios.append(audio)
        inputs, _ = self.model.featurizer(
            torch.tensor(audios[0]).to(self.device).unsqueeze(0),
            seq_len=torch.tensor(len(audios[0])).to(self.device).unsqueeze(0)
        )
        return {
            'input_features': inputs,
            'feature_attention_mask': torch.ones_like(inputs)[:,:,0]
        }

class WhisperPreprocess(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.audio_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3").feature_extractor
        self.audio_preprocessor = AudioPreprocessor(self.audio_processor)

    def __call__(self, audio_paths):
        audio_inputs_processed = self.audio_preprocessor.preprocess(audio_paths)
        inputs = {
            'input_features': audio_inputs_processed['input_features'],
            'feature_attention_mask': audio_inputs_processed['feature_attention_mask']
        }
        return inputs

# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------

if __name__=="__main__":
    # To use the processor (if available), set use_processor=True.
    similarity_fn = GlobalCosineSimilarity(model_type="wav2vec2", device="cpu", use_processor=True)
    score = similarity_fn("temp.wav", "temp.wav")
    print("Cosine similarity:", score)