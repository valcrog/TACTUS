"""High-level pipeline tying together feature extraction, cost, and alignment."""
from dataclasses import dataclass
import numpy as np

from .features import FeatureExtractor, ChromaSTFT
from .cost import compute_cost_matrix
from .aligners import Aligner, BasicDTW, AlignmentResult
from .visualization import plot_features, plot_alignment_path, plot_alignment_connections

@dataclass
class PipelineResult:
    """Full result of an alignment pipeline run."""
    X_features: np.ndarray
    Y_features: np.ndarray
    alignment: AlignmentResult
    sr: float
    hop_length: int

    def plot_features(self):
        plot_features(self.X_features, self.Y_features, self.hop_length, self.sr)

    def plot_alignment_path(self):
        plot_alignment_path(self.alignment.cost_matrix, self.alignment.path)

    def plot_alignment_connections(self):
        plot_alignment_connections(self.X_features, self.Y_features, self.alignment.path, self.hop_length)

    def plot(self):
        self.plot_features()
        self.plot_alignment_path()
        self.plot_alignment_connections()

class AlignmentPipeline:
    """A score-audio alignment pipeline.

    Args:
        feature_extractor (FeatureExtractor): Feature extraction strategy.
        aligner (Aligner): Alignment algorithm.
        metric (str): Distance metric for cost matrix computation.
    """
    def __init__(self, feature_extractor: FeatureExtractor = None, aligner: Aligner = None,
                 metric: str = 'euclidean'):
        self.feature_extractor = feature_extractor or ChromaSTFT()
        self.aligner = aligner or BasicDTW()
        self.metric = metric
        self._result = None

    @property
    def result(self) -> PipelineResult | None:
        """The result of the most recent pipeline run, or None if not run yet."""
        if self._result is None:
            print("Pipeline hasn't been run yet. No result available.")
        return self._result

    @property
    def hop_length(self) -> int:
        return self.feature_extractor.hop_length

    def run(self, x_wav: np.ndarray, y_wav: np.ndarray, sr: float = 22050) -> PipelineResult:
        """Run the full pipeline on two waveforms.

        Args:
            x_wav (np.ndarray): Reference (e.g. synthesized score) audio.
            y_wav (np.ndarray): Target (recorded performance) audio.
            sr (float): Sampling rate of both audio signals.

        Returns:
            PipelineResult
        """
        X = self.feature_extractor(x_wav, sr)
        Y = self.feature_extractor(y_wav, sr)
        C = compute_cost_matrix(X, Y, metric=self.metric)
        alignment = self.aligner.align(C)
        alignment.meta['sample_rate'] = sr

        result = PipelineResult(X, Y, alignment, sr, self.hop_length)
        self._result = result
        return result
