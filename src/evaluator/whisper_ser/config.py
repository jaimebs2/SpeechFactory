import torch
from .pooling import MeanPooling

class WhisperSERConfig:
    def __init__(
        self,
        projector_layers=[1280, 4],
        pooling=MeanPooling,
        embed_dim=1280,
        batch_norm=True,
        emo_dict={0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'},
        activation=torch.nn.GELU
    ):
        self.projector_layers = projector_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.emo_dict=emo_dict
