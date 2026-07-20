"""Feature extractors. Each is a callable class implementing `.extract(y, sr)`."""
from abc import ABC, abstractmethod

import numpy as np
import librosa

class FeatureExtractor(ABC):
    """Base class for feature extractors."""

    def __init__(self, hop_length=2205, n_fft=4410):
        self.hop_length = hop_length
        self.n_fft = n_fft

    @abstractmethod
    def extract(self, y, sr):
        """Compute a feature matrix of shape (n_features, n_frames)."""
        raise NotImplementedError

    def __call__(self, y, sr):
        return self.extract(y, sr)
    

class ChromaSTFT(FeatureExtractor):
    """Chroma features from an STFT."""
    def __init__(self, hop_length=2205, n_fft=4410, norm=2):
        super().__init__(hop_length, n_fft)
        self.norm = norm

    def extract(self, y, sr):
        return librosa.feature.chroma_stft(
            y=y, sr=sr, norm=self.norm,
            hop_length=self.hop_length, n_fft=self.n_fft
        )
    
class ChromaCQT(FeatureExtractor):
    """Chroma features from an CQT."""
    def __init__(self, hop_length=2205, norm=2):
        super().__init__(hop_length)
        self.norm = norm

    def extract(self, y, sr):
        return librosa.feature.chroma_cqt(
            y=y, sr=sr, norm=self.norm,
            hop_length=self.hop_length
        )

class ChromaCENS(FeatureExtractor):
    """CENS features (smoothed chroma)."""
    def __init__(self, hop_length=2205, n_fft=4410, win_len_smooth=41):
        super().__init__(hop_length, n_fft)
        self.win_len_smooth = win_len_smooth

    def extract(self, y, sr):
        return librosa.feature.chroma_cens(
            y=y, sr=sr, hop_length=self.hop_length,
            win_len_smooth=self.win_len_smooth
        )