from transformers import AutoModelForSpeechSeq2Seq
from torch import nn
import torch
from .utils import sinusoids, qwen2_attention_mask, whisper_attention_mask


class WhisperAvgEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(WhisperAvgEncoder, self).__init__()        

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3", use_safetensors=True
        ).model.encoder
        self.avg_pooler = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        torch.nn.init.constant_(self.model.layer_norm.bias, 0.0)
        torch.nn.init.constant_(self.model.layer_norm.weight, 1.0)

        self.register_buffer("positional_embedding", sinusoids(1500, 1280))
                
        self.audio_token_index = 151646
        self.pad_token_id = -1
        self.ignore_index = -100

    def encoder(self, x, feature_attention_mask):
        x = nn.functional.gelu(self.model.conv1(x))
        x = nn.functional.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        embed_pos = self.model.embed_positions.weight
        x = x + embed_pos

        for i, block in enumerate(self.model.layers):
            x = block(x, 
                    feature_attention_mask,
                    layer_head_mask=None,
                    output_attentions=False
                    )[0]
            #attentions[:, i, :, :, :] = block_attentions
        
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)

        x = self.model.layer_norm(x)
        return x, None #attentions

    def forward(self, x, feature_attention_mask):
        attention_mask, audio_output_lengths = qwen2_attention_mask(x, feature_attention_mask) # TODO: False to 0. and True to -inf
        hidden_states, attentions = self.encoder(
            x,
            attention_mask
        )

        return hidden_states, audio_output_lengths, attentions

class WhisperEncoder(nn.Module):
    def __init__(self):
        super(WhisperEncoder, self).__init__()        

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3", use_safetensors=True
        ).model.encoder
        self.avg_pooler = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        torch.nn.init.constant_(self.model.layer_norm.bias, 0.0)
        torch.nn.init.constant_(self.model.layer_norm.weight, 1.0)

        self.register_buffer("positional_embedding", sinusoids(1500, 1280))
                
        self.audio_token_index = 151646
        self.pad_token_id = -1
        self.ignore_index = -100

    def encoder(self, x, feature_attention_mask):
        x = nn.functional.gelu(self.model.conv1(x))
        x = nn.functional.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        embed_pos = self.model.embed_positions.weight
        x = x + embed_pos
        batch_size, seq_length, hidden_size = x.size()
        attentions = torch.empty(
            batch_size, 
            32, 
            20, 
            seq_length, 
            seq_length, 
            device=x.device, 
            dtype=x.dtype
        )
        for i, block in enumerate(self.model.layers):
            x, block_attentions = block(x, 
                    feature_attention_mask,
                    layer_head_mask=None,
                    output_attentions=True
                    )
            attentions[:, i, :, :, :] = block_attentions
    
        return x, attentions

    def forward(self, x, feature_attention_mask):
        attention_mask, audio_output_lengths = whisper_attention_mask(x, feature_attention_mask) # TODO: False to 0. and True to -inf
        hidden_states, attentions = self.encoder(
            x,
            attention_mask
        )

        return hidden_states, audio_output_lengths, attentions
