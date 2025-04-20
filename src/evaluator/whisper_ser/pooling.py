import torch
import torch.nn as nn

class BasePooling(nn.Module):
    def __init__(self, **kwargs):
        super(BasePooling, self).__init__()
        self.embed_dim = kwargs.get("embed_dim", 4096)

    def slice(self, *args, **kwargs):
        """
        A placeholder for slice operation.
        To be implemented by subclasses depending on the pooling strategy.
        """
        raise NotImplementedError("Subclasses must implement the 'slice' method.")

    def check_input_dim(self, x):
        """
        Ensures the input tensor has the correct embedding dimension.
        """
        if x.size(-1) != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embed_dim}, got {x.size(-1)}")

class MeanPooling(BasePooling):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def mean_pooling(self, x, attention_mask=None):
        if x.dim() != 4:
            x = x.unsqueeze(1)
        batch_size, num_channels, seq_length, embed_dim = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size * num_channels, seq_length)
            x = x.view(batch_size * num_channels, seq_length, embed_dim)
            masked_x = x * attention_mask.unsqueeze(-1)
            sum_masked_x = masked_x.sum(dim=1)
            sum_attention_mask = attention_mask.sum(dim=1).unsqueeze(-1)
            output = sum_masked_x / sum_attention_mask
            output = output.view(batch_size, num_channels, embed_dim)
        else:
            output = x.mean(dim=2)
        output[torch.isnan(output)] = 0
        attention_mask = torch.ones_like(output[:,:,0])
        return output.to(dtype=x.dtype), attention_mask
    
    def forward(self, x, mask):
        return self.mean_pooling(x, mask)
    
class AttentionMeanPooling(BasePooling):
    def __init__(self, embed_dim):
        super(AttentionMeanPooling, self).__init__()
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.apply(self._initialize_weights)

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
            torch.nn.init.constant_(layer.bias, 0)

    def compute_attention(self, x, attention_mask=None):
        if x.dim() != 4:
            x = x.unsqueeze(1)
        batch_size, num_channels, seq_length, embed_dim = x.size()
        x = x.view(batch_size * num_channels, seq_length, embed_dim)

        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (embed_dim ** 0.5)
        
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        output = torch.bmm(attn_weights, values).mean(dim=1, keepdim=True)
        attention_mask = torch.ones_like(output[:,:,0])
        return output, attention_mask

    def forward(self, x, mask):
        return self.compute_attention(x, mask)