import torch
from .config import WhisperSERConfig

class MLPProjector(torch.nn.Module):
    def __init__(self, precision=torch.bfloat16, config=WhisperSERConfig()):
        super(MLPProjector, self).__init__()
        layer_sizes = config.projector_layers
        batch_norm = config.batch_norm
        activation = config.activation
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False).to(dtype=precision))
            if i < len(layer_sizes) - 2:
                if batch_norm:
                    layers.append(torch.nn.LayerNorm(layer_sizes[i + 1],bias=False))
                layers.append(activation())
        
        self.model = torch.nn.Sequential(*layers)
        self.apply(self._initialize_weights)
        
    def forward(self, x):
        return self.model(x)
    
    def _initialize_weights(self, layer):            
        if isinstance(layer, torch.nn.Linear):
            if isinstance(layer, torch.nn.GELU) or isinstance(layer, torch.nn.ReLU):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            else:
                torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, torch.nn.LayerNorm):
            torch.nn.init.constant_(layer.weight, 1)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)
