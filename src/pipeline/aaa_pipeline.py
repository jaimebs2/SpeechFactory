import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from hyperopt import fmin, tpe, hp, Trials

import sys
sys.path.append(".")
from src.data.data_module import EmoDataset

from src.evaluator.ser import AttentionMeanPooling, WhisperSERConfig, EmotionClassifier

# ===========================
# Augmentation Operations
# ===========================

def add_noise(waveform, noise_level):
    """
    Adds Gaussian noise to the waveform.
    """
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def time_shift(waveform, shift_fraction):
    """
    Shifts the waveform by a fraction of its length.
    """
    shift_amount = int(shift_fraction * waveform.size(-1))
    return torch.roll(waveform, shifts=shift_amount, dims=-1)

def apply_waveform_augmentations(waveform, params):
    """
    Applies waveform-level augmentations (noise addition and time shift)
    based on probabilities and magnitudes defined in params.
    """
    # Apply noise addition with probability p_noise
    if random.random() < params['p_noise']:
        waveform = add_noise(waveform, params['m_noise'])
    # Apply time shift with probability p_shift
    if random.random() < params['p_shift']:
        waveform = time_shift(waveform, params['m_shift'])
    return waveform

def apply_spectrogram_augmentations(spec, params):
    """
    Applies spectrogram-level augmentations (time masking and frequency masking)
    using torchaudio transforms.
    """
    # Apply time masking with probability p_time_mask
    if random.random() < params['p_time_mask']:
        # Mask length proportional to m_time_mask * time dimension
        time_mask_param = max(1, int(params['m_time_mask'] * spec.size(-1)))
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        spec = time_mask(spec)
    # Apply frequency masking with probability p_freq_mask
    if random.random() < params['p_freq_mask']:
        # Mask length proportional to m_freq_mask * frequency dimension
        freq_mask_param = max(1, int(params['m_freq_mask'] * spec.size(-2)))
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        spec = freq_mask(spec)
    return spec

def mixup(x1, y1, x2, y2, alpha):
    """
    Applies a mixup operation between two samples.
    For simplicity in classification with CrossEntropyLoss, we select
    one label based on the mixing weight.
    """
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = y1 if lam > 0.5 else y2
    return mixed_x, mixed_y

# ===========================
# Composable Augmentation Policy
# ===========================

class AugmentationPolicy:
    def __init__(self, device, params):
        """
        Initializes the augmentation policy.
        The params dict should contain:
          - p_noise, m_noise: probability and magnitude for noise addition.
          - p_shift, m_shift: probability and fraction for time shift.
          - p_time_mask, m_time_mask: probability and magnitude for time masking.
          - p_freq_mask, m_freq_mask: probability and magnitude for frequency masking.
          - p_mix, m_mix: probability for mixup and the alpha parameter for Beta distribution.
        """
        self.params = params
        # MelSpectrogram transform for converting waveform to spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64).to(device)

    def apply(self, waveform, label, other_waveform=None, other_label=None):
        """
        Applies the composed augmentation:
          1. Waveform-level augmentations.
          2. Conversion to spectrogram.
          3. Spectrogram-level augmentations.
          4. Optional mixup with a second sample.
        """
        # Waveform-level augmentation
        waveform_aug = apply_waveform_augmentations(waveform, self.params)
        # Convert augmented waveform to spectrogram
        spec = self.mel_transform(waveform_aug)
        # Spectrogram-level augmentation
        spec_aug = apply_spectrogram_augmentations(spec, self.params)
        # Mixed augmentation: if a second sample is provided and condition met
        if other_waveform is not None and random.random() < self.params['p_mix']:
            other_waveform_aug = apply_waveform_augmentations(other_waveform, self.params)
            other_spec = self.mel_transform(other_waveform_aug)
            other_spec_aug = apply_spectrogram_augmentations(other_spec, self.params)
            spec_aug, label = mixup(spec_aug, label, other_spec_aug, other_label, self.params['m_mix'])
        return spec_aug, label

# ===========================
# Training and Evaluation Functions
# ===========================

def train_one_epoch(model, optimizer, criterion, dataloader, augmentation_policy, device):
    model.train()
    running_loss = 0.0
    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        # Convert labels to tensor
        labels = torch.tensor(labels).to(device)
        batch_size = waveforms.size(0)
        augmented_specs = []
        augmented_labels = []
        # For each sample in the batch, apply augmentation.
        # For mixup, we randomly select a partner sample from the batch.
        for j in range(batch_size):
            other_idx = random.randint(0, batch_size - 1)
            if other_idx == j:
                other_idx = (other_idx + 1) % batch_size
            spec, lbl = augmentation_policy.apply(waveforms[j], labels[j],
                                                  waveforms[other_idx], labels[other_idx])
            augmented_specs.append(spec)
            augmented_labels.append(lbl)
        specs = torch.stack(augmented_specs)  # (batch, channels, height, width)
        # Ensure there is an explicit channel dimension
        if specs.dim() == 3:
            specs = specs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, torch.tensor(augmented_labels).to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64).to(device)
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = torch.tensor(labels).to(device)
            specs = []
            for j in range(waveforms.size(0)):
                spec = mel_transform(waveforms[j])
                specs.append(spec)
            specs = torch.stack(specs)
            if specs.dim() == 3:
                specs = specs.unsqueeze(1)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# ===========================
# Policy Search Using HyperOpt
# ===========================

def objective(params, train_subset, val_subset, device):
    """
    The objective function for HyperOpt.
    A new augmentation policy is built from candidate parameters;
    a simple model is trained for a few epochs on a small subset,
    and negative validation accuracy is returned as the loss.
    """
    aug_policy = AugmentationPolicy(device, params)
    model = EmotionClassifier(
        config=WhisperSERConfig(
            projector_layers=[1280, 4],
            batch_norm=True,
            pooling=AttentionMeanPooling,
            embed_dim=1280,
            emo_dict={0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'},
            activation=torch.nn.GELU
        ),
        device=device,
        dtype=torch.bfloat16, 
        model_path="models_weights/whisper-ser-iemocap_meacorpus/whisper-ser-iemocap_meacorpus.ckpt"
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=16, shuffle=False)
    # Train for 2 epochs (short evaluation)
    for epoch in range(2):
        train_one_epoch(model, optimizer, criterion, train_loader, aug_policy, device)
    _, val_acc = validate(model, criterion, val_loader, device)
    return -val_acc  # HyperOpt minimizes, so we return negative accuracy

def policy_search(train_subset, val_subset, device, max_evals=10):
    """
    Searches for better augmentation policy parameters using HyperOpt.
    The search space here is a simplified version.
    """
    space = {
        'p_noise': hp.uniform('p_noise', 0.0, 1.0),
        'm_noise': hp.uniform('m_noise', 0.0, 0.5),
        'p_shift': hp.uniform('p_shift', 0.0, 1.0),
        'm_shift': hp.uniform('m_shift', 0.0, 0.2),
        'p_time_mask': hp.uniform('p_time_mask', 0.0, 1.0),
        'm_time_mask': hp.uniform('m_time_mask', 0.0, 0.5),
        'p_freq_mask': hp.uniform('p_freq_mask', 0.0, 1.0),
        'm_freq_mask': hp.uniform('m_freq_mask', 0.0, 0.5),
        'p_mix': hp.uniform('p_mix', 0.0, 1.0),
        'm_mix': hp.uniform('m_mix', 0.1, 0.5)
    }
    trials = Trials()
    best = fmin(lambda params: objective(params, train_subset, val_subset, device),
                space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best

# ===========================
# Main Training Loop with Policy Updates
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create dummy dataset
    train_loader = EmoDataset(
            target='meacorpus',
            reference='meacorpus',
            fold=1,
            language_code='es',
            device=device,
            denoise=False,
            skip=0,
            audio_description=False,
            split='train',
        )
    
    val_loader = EmoDataset(
            target='meacorpus',
            reference='meacorpus',
            fold=1,
            language_code='es',
            device=device,
            denoise=False,
            skip=0,
            audio_description=False,
            split='valid',
        )
    
    # Initial augmentation policy parameters (random initialization)
    init_params = {
        'p_noise': 0.5,
        'm_noise': 0.1,
        'p_shift': 0.5,
        'm_shift': 0.05,
        'p_time_mask': 0.5,
        'm_time_mask': 0.2,
        'p_freq_mask': 0.5,
        'm_freq_mask': 0.2,
        'p_mix': 0.5,
        'm_mix': 0.3
    }
    aug_policy = AugmentationPolicy(device, init_params)
    
    # Initialize model, optimizer, and loss function
    model = EmotionClassifier(
                        config=WhisperSERConfig(
                            projector_layers=[1280, 4],
                            batch_norm=True,
                            pooling=AttentionMeanPooling,
                            embed_dim=1280,
                            emo_dict={0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'},
                            activation=torch.nn.GELU
                        ),
                        device=device,
                        dtype=torch.bfloat16, 
                        model_path="models_weights/whisper-ser-iemocap_meacorpus/whisper-ser-iemocap_meacorpus.ckpt"
                    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Outer loop: alternating training and policy search updates
    T_update = 3  # number of policy update rounds
    T_train = 5   # epochs per update round
    for update in range(T_update):
        print(f"\n--- Outer Loop Update {update+1}/{T_update} ---")
        # Inner training loop
        for epoch in range(T_train):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, aug_policy, device)
            val_loss, val_acc = validate(model, criterion, val_loader, device)
            print(f"Epoch {epoch+1}/{T_train}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        # Policy search update: use small subsets for fast evaluation
        train_subset = torch.utils.data.Subset(train_loader.target_data, list(range(min(50, len(train_loader.target_data)))))
        val_subset = torch.utils.data.Subset(val_loader.target_data, list(range(min(20, len(val_loader.target_data)))))
        best_params = policy_search(train_subset, val_subset, device, max_evals=10)
        print("Updated policy parameters:", best_params)
        # Update the augmentation policy with the new parameters
        aug_policy = AugmentationPolicy(device, best_params)
    
    # Final evaluation on the validation set
    final_loss, final_acc = validate(model, criterion, val_loader, device)
    print(f"\nFinal Evaluation: Val Loss = {final_loss:.4f}, Val Acc = {final_acc:.4f}")

if __name__ == "__main__":
    main()
