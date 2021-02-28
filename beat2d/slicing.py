from typing import List, Tuple

import numpy as np
import librosa

from beat2d import settings


def get_onsets(
    sample: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onset_envelope = librosa.onset.onset_strength(sample, sr=sr)

    onsets = librosa.util.peak_pick(onset_envelope, 10, 10, 10, 10, 0.25, 5)
    onsets = librosa.onset.onset_backtrack(onsets, onset_envelope)

    return onsets


def calc_slice_points(sample: np.ndarray, sr: int = settings.SAMPLE_RATE) -> List[Tuple[int, int]]:
    """Return a list of tuples of slice points in samples."""
    slice_points: List[Tuple[int, int]] = []

    # Calculate the onsets
    onsets = get_onsets(sample, sr)

    # Convert onset times to samples
    onset_samples = librosa.frames_to_samples(onsets)

    # Loop over each onset to the second last one
    for idx, _ in enumerate(onset_samples[:-1]):
        slice_pair: Tuple[int, int] = (onset_samples[idx], onset_samples[idx + 1])
        slice_points.append(slice_pair)

    # Need one slice point from the last onset to the end of the audio
    slice_points.append((onset_samples[-1], len(sample) - 1))

    return slice_points


def split_oneshots(sample: np.ndarray, sr: int = settings.SAMPLE_RATE) -> np.ndarray:
    # Get slice points
    slice_points: List[Tuple[int, int]] = calc_slice_points(sample, sr)

    # Slice sample into separate samples by onsets
    slices: List[np.ndarray] = []

    # The first sample starts at the first onset
    for (slice_start, slice_end) in slice_points:
        slice_ = sample[slice_start:slice_end]
        slices.append(slice_)

    return slices
