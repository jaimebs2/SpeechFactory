import torch
import numpy as np

def whisper_attention_mask(x, feature_attention_mask):
    audio_feat_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1

    batch_size, _, max_mel_seq_len = x.shape
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1

    seq_range = (
        torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
        .unsqueeze(0)
        .expand(batch_size, max_seq_len)
    )
    lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
    
    padding_mask = seq_range >= lengths_expand

    audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
        batch_size, 1, max_seq_len, max_seq_len
    )
    
    audio_attention_mask = audio_attention_mask.float()
    audio_attention_mask[audio_attention_mask == 1.] = float("-inf")
    return audio_attention_mask.to(dtype=torch.bfloat16), audio_feat_lengths

def qwen2_attention_mask(x, feature_attention_mask):
    audio_feat_lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1
    audio_output_lengths = (audio_feat_lengths - 2) // 2 + 1
    batch_size, _, max_mel_seq_len = x.shape
    max_seq_len = (max_mel_seq_len - 2) // 2 + 1

    seq_range = (
        torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
        .unsqueeze(0)
        .expand(batch_size, max_seq_len)
    )
    lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
    
    padding_mask = seq_range >= lengths_expand

    audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
        batch_size, 1, max_seq_len, max_seq_len
    )
    
    audio_attention_mask = audio_attention_mask.float()
    audio_attention_mask[audio_attention_mask == 1.] = float("-inf")
    return audio_attention_mask.to(dtype=torch.bfloat16), audio_output_lengths

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def audio_features_pad(audio_features, features_mask):
    """
    Pads the audio features after applying the mask and outputs the padded audio features along with the new features mask.

    Args:
        audio_features (torch.Tensor): Tensor of shape [batch_size, seq_length, embed_dim].
        features_mask (torch.Tensor): Tensor of shape [batch_size, seq_length], boolean mask indicating valid positions.

    Returns:
        padded_audio_features (torch.Tensor): Tensor of shape [batch_size, max_seq_length, embed_dim], padded features.
        new_features_mask (torch.Tensor): Tensor of shape [batch_size, max_seq_length], mask for the padded features.
    """

    batch_size, seq_length, embed_dim = audio_features.size()

    features_list = []
    lengths = []

    for i in range(batch_size):
        # Apply the mask to remove unused features
        mask_i = features_mask[i].bool()  # Shape: [seq_length]
        audio_features_i = audio_features[i][mask_i]  # Shape: [remaining_seq_length, embed_dim]
        features_list.append(audio_features_i)
        lengths.append(audio_features_i.size(0))

    # Find the maximum sequence length after masking
    max_length = max(lengths)

    # Prepare tensors to hold the padded features and masks
    padded_audio_features = torch.zeros(
        batch_size, max_length, embed_dim,
        dtype=audio_features.dtype, device=audio_features.device
    )
    new_features_mask = torch.zeros(
        batch_size, max_length,
        dtype=features_mask.dtype, device=features_mask.device
    )

    # Pad each audio_features to have the same sequence length
    for i, (features_i, length_i) in enumerate(zip(features_list, lengths)):
        padded_audio_features[i, :length_i, :] = features_i
        new_features_mask[i, :length_i] = 1  # Set mask to 1 for valid positions

    return padded_audio_features, new_features_mask


def audio_features_mask(audio_features, num_audio_tokens):
    if num_audio_tokens.numel() == 1:
        return torch.ones(audio_features.shape[:2], dtype=torch.int16, device=audio_features.device)
    if num_audio_tokens.dim() == 2:
        num_audio_tokens = num_audio_tokens.squeeze(1)

    _, seq_length, _ = audio_features.size()
    positions = torch.arange(seq_length).unsqueeze(0)
    attention_mask = (positions.to(num_audio_tokens.device) < num_audio_tokens.unsqueeze(1)).long() 
    return attention_mask