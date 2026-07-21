"""Input/Output utilities: score loading, MIDI synthesis, audio loading."""

import os
from typing import Tuple

import numpy as np
import music21 as m21
import librosa
import pretty_midi
import tinysoundfont

from .aligners import AlignmentResult

def load_score(path: str, format: str = None) -> m21.stream.Score:
    """Parse the given file as a music21 `Score`

    Args:
        path (str): the file path
        format (str, optional): the format to parse the line of text or the file as. If not given, \
        guess the format from the file extension.

    Raises:
        ValueError: if the given path doesn't exist

    Returns:
        m21.stream.Score: the stream parsed from the given file
    """
    if not os.path.exists(path):
        raise ValueError(f"Couldn't load score : File {path} doesn't exist")
    
    return m21.converter.parse(path)

def get_midi_object(score: m21.stream.Score) -> pretty_midi.PrettyMIDI:
    """Return the pretty_midi MIDI object corresponding to the score

    Args:
        score (m21.stream.Score): the score to convert to MIDI

    Returns:
        pretty_midi.PrettyMIDI: a MIDI object
    """
    fp = score.write('midi')

    return pretty_midi.PrettyMIDI(fp)

def synthesize_midi(midi_path, sr: float = 22050, soundfont="data/TimGM6mb.sf2") -> np.ndarray:
    synth = tinysoundfont.Synth(samplerate=sr)
    sfid = synth.sfload(soundfont)

    for chan in range(16):
        synth.program_select(chan, sfid, 0, 0)

    seq = tinysoundfont.Sequencer(synth)
    seq.midi_load(midi_path)

    chunks = []
    while not seq.is_empty():
        chunks.append(synth.generate(1024).copy())
    
    chunks.append(synth.generate(int(sr)).copy()) # release for the last notes

    audio = np.concatenate(chunks)
    return audio.mean(axis=1)

def synthesize_score(score: m21.stream.Score, sr: float = 22050, soundfont="data/TimGM6mb.sf2") -> Tuple[np.ndarray, int | float]:
    """Synthesize a given music21 score to a floating point time series using fluidsynth

    Args:
        score (m21.stream.Score): the score to synthesize
        sr (float, optional): sampling rate to synthesize at. Defaults to 22050.
        soundfont (str, optional): path to the soundfont used for synthesis. Defaults to Grand Piano.

    Returns:
        a tuple (y, sr) containing
        - y (np.ndarray) : audio time series.
        - sr (number > 0 [scalar]) : sampling rate of ``y``
    """
    fp = score.write('midi')

    return (synthesize_midi(fp, sr, soundfont), sr)

def trim_silence(y: np.ndarray, threshold: float = 30) -> np.ndarray:
    """Trim leading and trailing silence from the audio signal

    Args:
        y (np.ndarray): the audio signal to trim
        threshold (float, optional): the threshold (in dB) below reference to consider as silence. Defaults to 60.

    Returns:
        np.ndarray: the trimmed signal
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=threshold)

    return y_trimmed

def load_audio(path: str, sr: float = 22050, duration: float = None, trim_threshold: float = None) -> Tuple[np.ndarray, int | float]:
    """Load an audio file as a floating point time series

    Args:
        path (str): the file path
        sr (float, optional): the desired sample rate. To preserve the native sampling \
                rate of the file, use `sr=None`. Defaults to 22050.
        duration (float, optional): only load up to this much audio (in seconds). Defaults to None.
        trim_threshold (float, optional): if given, trim the leading and trailing silence \
                below this threshold (in dB). Defaults to None.
    Returns:
        a tuple (y, sr) containing
        - y (np.ndarray [shape=(n,) or (..., n)]) : audio time series. Multi-channel is supported.
        - sr (number > 0 [scalar]) : sampling rate of ``y``
    """
    y, fs = librosa.load(path, 
                         sr=sr, 
                         duration=duration)

    if trim_threshold is not None:
        y = trim_silence(y, threshold=trim_threshold)

    return (y, fs)

class AlignmentExporter:
    def __init__(self, result: AlignmentResult):
        self.result = result