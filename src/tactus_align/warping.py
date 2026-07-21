"""Path post-processing and audio warping utilities."""
import numpy as np
from scipy.interpolate import interp1d
import music21 as m21
import pretty_midi
from librosa import frames_to_time

from .pipeline import PipelineResult
from .io import get_midi_object, synthesize_midi

def path_to_time_map(path, hop_length, sr):
    """Convert a frame-wise warping path into (t_X, t_Y) pairs in seconds.

    Args:
        path (np.ndarray): Warping path (L, 2).
        hop_length (int): Hop size used for feature extraction.
        sr (int): Sampling rate.

    Returns:
        (np.ndarray, np.ndarray): Times in seconds for X and Y respectively.
    """
    t_X = path[:, 0] * hop_length / sr
    t_Y = path[:, 1] * hop_length / sr
    return t_X, t_Y


def warp_audio(x, y, path, hop_length, mode='X_to_Y'):
    """Time-warp one signal onto the timeline of the other using the warping path.

    Args:
        x (np.ndarray): First signal (corresponds to path axis 0).
        y (np.ndarray): Second signal (corresponds to path axis 1).
        path (np.ndarray): Warping path (L, 2).
        hop_length (int): Feature hop length in samples.
        mode (str): Either 'X_to_Y' (default: map x onto y's timeline) or 'Y_to_X'.

    Returns:
        np.ndarray: Warped signal, same length as the target signal.
    """
    x_anchors = (path[:, 0] * hop_length).astype(float)
    y_anchors = (path[:, 1] * hop_length).astype(float)

    if mode == 'X_to_Y':
        anchors_in, anchors_out = y_anchors, x_anchors
        target_len, src = len(y), x
    elif mode == 'Y_to_X':
        anchors_in, anchors_out = x_anchors, y_anchors
        target_len, src = len(x), y
    else:
        raise ValueError("mode must be 'X_to_Y' or 'Y_to_X'")

    _, unique_idx = np.unique(anchors_in, return_index=True)
    ai = anchors_in[unique_idx]
    ao = anchors_out[unique_idx]

    mapper = interp1d(ai, ao, kind='linear',
                      bounds_error=False, fill_value=(ao[0], ao[-1]))
    t_indices = mapper(np.arange(target_len))
    t_indices = np.clip(t_indices, 0, len(src) - 1)

    i0 = np.floor(t_indices).astype(int)
    i1 = np.clip(i0 + 1, 0, len(src) - 1)
    frac = t_indices - i0
    return (1 - frac) * src[i0] + frac * src[i1]

def warp_midi(score: m21.stream.Score, result: PipelineResult, mode: str = 'X_to_Y') -> pretty_midi.PrettyMIDI:
    """Warp the score's MIDI representation using an alignment result

    Args:
        score (m21.stream.Score): the score to warp
        result (PipelineResult): the alignment result containing the warping path and feature parameters
        mode (str, optional): Either 'X_to_Y' (default: map score x onto audio y's timeline) or 'Y_to_X'. Defaults to 'X_to_Y'.

    Raises:
        ValueError: _description_

    Returns:
        pretty_midi.PrettyMIDI: a MIDI object with adjusted note timings according to the warping path.
    """
    midi_object = get_midi_object(score)
    
    if mode == 'Y_to_X':
        X = result.Y_features
        Y = result.X_features
        p = result.alignment.path[:, 1]
        q = result.alignment.path[:, 0]
    elif mode == 'X_to_Y':
        X = result.X_features
        Y = result.Y_features
        p = result.alignment.path[:, 0]
        q = result.alignment.path[:, 1]
    else:
        raise ValueError(f'Unknown mode {mode} (must be "X_to_Y" or "Y_to_X") : see documentation for more details')

    midi_times = frames_to_time(np.arange(np.size(X, axis=1)), 
                                sr=result.sr, 
                                hop_length=result.hop_length)
    audio_times = frames_to_time(np.arange(np.size(Y, axis=1)),
                                 sr=result.sr,
                                 hop_length=result.hop_length)
    
    midi_object.adjust_times(midi_times[p], audio_times[q])

    return midi_object

def warp_synth(score: m21.stream.Score, result: PipelineResult, mode: str = 'X_to_Y') -> np.ndarray:
    """Warp the score's MIDI representation according to the alignment result,
    then synthetizes it using the same parameters

    Args:
        score (m21.stream.Score): the score used to get the alignment result
        result (PipelineResult): the alignment result
        mode (str): Either 'X_to_Y' (default: map score x onto audio y's timeline) or 'Y_to_X'. \
            By default, the score is the first signal (path axis 0) and the audio is the second (axis 1). \
            If the first signal is the original audio, select mode `Y_to_X`

    Returns:
        np.ndarray: an audio signal generated from warped MIDI.
    """

    midi_object = warp_midi(score, result, mode=mode)
    fp = midi_object.write("temp.mid")
    X_warped = synthesize_midi(fp, result.sr)

    return X_warped

def stereo_mix(a, b):
    """Normalize and stack two signals as a stereo array for side-by-side playback."""
    def _norm(s):
        m = np.max(np.abs(s))
        return s / m * 0.9 if m > 0 else s
    
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    
    return np.vstack([_norm(a), _norm(b)])
