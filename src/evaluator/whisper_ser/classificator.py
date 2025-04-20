import torch
import torch.nn as nn
import numpy as np

from .projector import MLPProjector
from .whisper_encoder import WhisperEncoder
from .pooling import MeanPooling
from .utils import audio_features_mask, audio_features_pad
from .config import WhisperSERConfig

class Classificator(nn.Module):
    def __init__(self, config=WhisperSERConfig()):
        super().__init__()
        self.config = config

        self.audio_encoder = WhisperEncoder()
        self.projector = MLPProjector(config = self.config)
        self.pooling = config.pooling(config.embed_dim)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.emo_dict = config.emo_dict

    def forward(
        self,
        audio_features,
        audio_len=None,
    ):
        batch_size = audio_features.size(0)

        audio_attention_mask = torch.zeros(batch_size, audio_features.size(2), device=audio_features.device)
        factor = np.round(audio_features.size(-1) / max(audio_len).item())

        if audio_len.numel() == 1:
            audio_attention_mask[0, :int(audio_len.item() * factor)] = 1
        else:
            for i in range(batch_size):
                audio_attention_mask[i, :int(audio_len[i].item() * factor)] = 1

        audio_features, audio_len, _ = self.audio_encoder(audio_features, audio_attention_mask)

        audio_mask = audio_features_mask(audio_features, audio_len)
        audio_features, new_features_mask = audio_features_pad(audio_features, audio_mask)

        audio_features, new_features_mask = self.pooling(
            audio_features, new_features_mask
        )

        outputs = self.projector(audio_features.squeeze(1))

        return outputs, audio_features.squeeze(1)
    
    def predict(self, audio_features, audio_len):
        with torch.no_grad():
            probs, features = self.forward(audio_features, audio_len)
            preds = torch.argmax(probs, dim=1)
            preds = self.emo_dict[preds.item()]
        return preds, probs, features
