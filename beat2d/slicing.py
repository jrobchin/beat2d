from typing import List, Tuple

import numpy as np
import librosa

from beat2d import settings


def calc_slice_points(sample: np.ndarray, sr: int = settings.SAMPLE_RATE) -> List[Tuple[int, int]]:
    """Return a list of tuples of slice points in samples."""
    slice_points: List[Tuple[int, int]] = []

    # Calculate the onsets
    onsets = librosa.onset.onset_detect(
        sample,
        sr,
        backtrack=True,
        units="samples",
        pre_max=10,
        post_max=1,
        pre_avg=3,
        post_avg=1,
        delta=0.2,
        wait=10,
    )

    # Loop over each onset to the second last one
    for idx, _ in enumerate(onsets[:-1]):
        slice_pair: Tuple[int, int] = (onsets[idx], onsets[idx + 1])
        slice_points.append(slice_pair)

    # Need one slice point from the last onset to the end of the audio
    slice_points.append((onsets[-1], len(sample) - 1))

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
