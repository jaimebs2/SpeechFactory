import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

class FishSpeechWrapper:
    def __init__(self, checkpoint_path, device="cuda"):
        """Initialize the Fish Speech model with path fixing"""
        # Add paths to ensure imports work
        repo_root = Path('/home/jaime/repos')
        fish_speech_path = repo_root / 'SpeechFactory' / 'src' / 'generator' / 'fish_speech'
        
        if str(fish_speech_path) not in sys.path:
            sys.path.append(str(fish_speech_path))
            
        # Import from correct path
        from fish_speech.models.vqgan.inference import load_model
        
        # Load vocoder model
        self.model = self.load_model_fixed_paths(checkpoint_path, device)
        self.device = device
        
        # Initialize text-to-semantic model
        self.llm_model = None
        self.decode_one_token = None
        
    def load_text_to_semantic_model(self, llm_checkpoint_path):
        """Load text-to-semantic model"""
        from fish_speech.models.text2semantic.inference import load_model as load_llm
        
        print(f"Loading text-to-semantic model from {llm_checkpoint_path}")
        precision = torch.half if torch.cuda.is_available() else torch.bfloat16
        self.llm_model, self.decode_one_token = load_llm(
            llm_checkpoint_path, 
            self.device, 
            precision,
            compile=False
        )
        
        # Setup model caches
        with torch.device(self.device):
            self.llm_model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.llm_model.config.max_seq_len,
                dtype=next(self.llm_model.parameters()).dtype,
            )
        
        print("Text-to-semantic model loaded successfully")
        return self.llm_model
        
    def load_model_fixed_paths(self, checkpoint_path, device="cuda"):
        """Custom model loader that fixes module paths in config"""
        import hydra
        from hydra import compose, initialize
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import torch
        from loguru import logger
        
        # Clear any existing Hydra config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        # Register eval resolver if needed
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        
        # Load the config
        with initialize(version_base="1.3", config_path=str(Path('fish_speech/fish_speech/configs'))):
            cfg = compose(config_name="firefly_gan_vq")
        
        # Fix module paths by editing config 
        # Remove duplicate 'vqgan' in paths and adjust as needed
        def fix_path(path_str):
            """Fix module paths in config"""
            # Example: Convert 
            # "fish_speech.models.vqgan.modules.vqgan.modules.firefly.ConvNeXtEncoder"
            # to
            # "fish_speech.models.vqgan.modules.firefly.ConvNeXtEncoder"
            return path_str.replace("vqgan.modules.vqgan.modules", "vqgan.modules")
        
        # Fix backbone path
        if "_target_" in cfg.backbone:
            cfg.backbone._target_ = fix_path(cfg.backbone._target_)
        
        # Fix head path
        if "_target_" in cfg.head:
            cfg.head._target_ = fix_path(cfg.head._target_)
        
        # Fix quantizer path
        if "_target_" in cfg.quantizer:
            cfg.quantizer._target_ = fix_path(cfg.quantizer._target_)
        
        print(f"Using fixed module paths:")
        print(f"Backbone: {cfg.backbone._target_}")
        print(f"Head: {cfg.head._target_}")
        print(f"Quantizer: {cfg.quantizer._target_}")
        
        # Instantiate model with fixed paths
        model = instantiate(cfg)
        
        # Load weights
        state_dict = torch.load(
            checkpoint_path, map_location=device, mmap=True, weights_only=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        model.eval()
        model.to(device)

        return model
    
    def encode(self, audio, sr):
        """Encode audio to indices"""
        # Ensure audio format and sample rate
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() > 2:
            audio = audio.squeeze(0)
        if audio.shape[0] > 1:  # Convert stereo to mono
            audio = audio.mean(0, keepdim=True)
        
        # Resample if needed
        if sr != self.model.spec_transform.sample_rate:
            audio = torchaudio.functional.resample(
                audio, sr, self.model.spec_transform.sample_rate
            )
        
        # Move to device
        audio = audio.to(self.device)
        
        # Encode
        with torch.no_grad():
            audio_lengths = torch.tensor([audio.shape[1]], device=self.device, dtype=torch.long)
            indices = self.model.encode(audio.unsqueeze(0), audio_lengths)[0][0]
        
        return indices
    
    def decode(self, indices):
        """Decode indices to audio"""
        with torch.no_grad():
            feature_lengths = torch.tensor([indices.shape[1]], device=self.device)
            fake_audios, _ = self.model.decode(
                indices=indices.unsqueeze(0), 
                feature_lengths=feature_lengths
            )
        
        # Return the audio and sample rate
        return fake_audios[0, 0].float().cpu().numpy(), self.model.spec_transform.sample_rate
    
    def generate_from_text(self, text, prompt_tokens, num_samples=1, 
                          temperature=0.7, top_p=0.7, repetition_penalty=1.2):
        """Generate semantic tokens from text using reference voice prompt"""
        if self.llm_model is None:
            raise ValueError("Text-to-semantic model not loaded. Call load_text_to_semantic_model first.")
            
        from fish_speech.models.text2semantic.inference import generate_long, GenerateResponse
        
        # Set up parameters
        kwargs = {
            "device": self.device,
            "decode_one_token": self.decode_one_token,
            "text": text,
            "num_samples": num_samples,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "compile": False,
            "prompt_text": ["Reference voice"],  # Description for prompt
            "prompt_tokens": [prompt_tokens],    # Actual voice tokens
        }
        
        # Generate semantic tokens
        generator = generate_long(model=self.llm_model, **kwargs)
        
        semantic_codes = []
        for response in generator:
            if response.action == "sample":
                semantic_codes.append(response.codes)
            elif response.action == "next":
                if len(semantic_codes) > 0:
                    # We only care about the first sample for now
                    break
        
        if not semantic_codes:
            raise RuntimeError("Failed to generate semantic tokens from text")
            
        # Concatenate all codes for the text
        return torch.cat(semantic_codes, dim=1)
    
    def clone_voice(self, reference_audio, reference_sr, text, temperature=0.7, top_p=0.7):
        """Complete voice cloning pipeline: reference audio → text → generated audio"""
        # 1. Encode reference audio to get voice prompt
        prompt_indices = self.encode(reference_audio, reference_sr)
        
        # 2. Generate semantic tokens from text using the reference voice prompt
        semantic_indices = self.generate_from_text(
            text, prompt_indices, 
            temperature=temperature, 
            top_p=top_p
        )
        
        # 3. Decode semantic tokens to audio
        audio, sample_rate = self.decode(semantic_indices)
        
        return audio, sample_rate