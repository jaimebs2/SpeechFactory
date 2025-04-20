import numpy as np
import torch
import torchaudio
import librosa
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

class FeatureExtractor(ABC):
    """Base class for audio feature extractors"""
    
    @abstractmethod
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from an audio file"""
        pass

class MFCCExtractor(FeatureExtractor):
    """MFCC feature extractor using librosa"""
    
    def __init__(self, n_mfcc: int = 13, sr: int = 22050):
        self.n_mfcc = n_mfcc
        self.sr = sr
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio file"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        # Flatten and return mean of each coefficient over time
        return np.mean(mfccs, axis=1)

class MelSpectrogramExtractor(FeatureExtractor):
    """Mel spectrogram feature extractor using librosa"""
    
    def __init__(self, n_mels: int = 128, sr: int = 22050):
        self.n_mels = n_mels
        self.sr = sr
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract mel spectrogram features from audio file"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        # Convert to log scale and compute mean over time
        log_mel_spec = librosa.power_to_db(mel_spec)
        return np.mean(log_mel_spec, axis=1)

class VGGishExtractor(FeatureExtractor):
    """VGGish feature extractor using torchaudio"""
    
    def __init__(self):
        try:
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.model.eval()
        except Exception as e:
            raise ImportError(f"Could not load VGGish model: {e}")
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract VGGish embeddings from audio file"""
        with torch.no_grad():
            embedding = self.model.forward(audio_path)
        return embedding.numpy()

class CosineSimilarity:
    """Class to compute cosine similarity between two audio files"""
    
    def __init__(self, model: str = "mfcc"):
        """
        Initialize with the chosen feature extractor model
        
        Args:
            model (str): Model to use for feature extraction.
                         Options: "mfcc", "melspectrogram", "vggish",
                                  "hubert", "whisper", "canary", 
                                  "wav2vec2", "ecapa", "xvector"
        """
        self.extractor = self._get_extractor(model)
    
    def _get_extractor(self, model: str) -> FeatureExtractor:
        """Get the appropriate feature extractor based on the model name"""
        if model.lower() == "mfcc":
            return MFCCExtractor()
        elif model.lower() == "melspectrogram":
            return MelSpectrogramExtractor()
        elif model.lower() == "vggish":
            return VGGishExtractor()
        elif model.lower() == "hubert":
            from .embedders import HubertFeatureExtractor
            return HubertFeatureExtractor()
        elif model.lower() == "whisper":
            from .embedders import WhisperFeatureExtractor
            return WhisperFeatureExtractor()
        elif model.lower() == "canary":
            from .embedders import CanaryFeatureExtractor
            return CanaryFeatureExtractor()
        elif model.lower() == "wav2vec2":
            from .embedders import Wav2Vec2FeatureExtractor
            return Wav2Vec2FeatureExtractor()
        elif model.lower() == "ecapa":
            from .embedders import EcapaFeatureExtractor
            return EcapaFeatureExtractor()
        elif model.lower() == "xvector":
            from .embedders import XVectorFeatureExtractor
            return XVectorFeatureExtractor()
        else:
            raise ValueError(f"Unknown model: {model}. Available models: mfcc, melspectrogram, vggish, hubert, whisper, canary, wav2vec2, ecapa, xvector")
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        # Calculate norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def calculate_similarity(self, audio_path1: str, audio_path2: str) -> float:
        """
        Calculate cosine similarity between two audio files
        
        Args:
            audio_path1 (str): Path to first audio file
            audio_path2 (str): Path to second audio file
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Extract features from both audio files
        features1 = self.extractor.extract_features(audio_path1)
        features2 = self.extractor.extract_features(audio_path2)
        
        # Ensure features have the same dimensions
        if features1.shape != features2.shape:
            raise ValueError(f"Feature shapes don't match: {features1.shape} vs {features2.shape}")
        
        # Compute cosine similarity
        similarity = self._compute_cosine_similarity(features1, features2)
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))

    def __call__(self, audio_path1: str, audio_path2: str) -> float:
        """
        Callable interface for calculating similarity
        
        Args:
            audio_path1 (str): Path to first audio file
            audio_path2 (str): Path to second audio file
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        return self.calculate_similarity(audio_path1, audio_path2)
