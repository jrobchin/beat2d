from typing import List, Tuple

import numpy as np
import librosa
from librosa import core
from librosa import ParameterError
from librosa.onset import onset_strength, onset_backtrack
from librosa import util

from beat2d import settings


def onset_detect(
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    backtrack=False,
    energy=None,
    units="frames",
    normalize=True,
    **kwargs
):
    """Basic onset detector.  Locate note onset events by picking peaks in an
    onset strength envelope. Modified from `librosa.onset.onset_detect` to add a
    `normalize` flag.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [1]_.

    .. [1] https://github.com/CPJKU/onset_db


    Parameters
    ----------
    y          : np.ndarray [shape=(n,)]
        audio time series

    sr         : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope     : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If `True`, detected onset events are backtracked to the nearest
        preceding minimum of `energy`.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then `onset_envelope` is used.

    noramlize : bool (optional)
        If `True`, normalize the onset envelope before peak picking. By default
        this parameter is `True`.

    kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.


    Returns
    -------

    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.


    Raises
    ------
    ParameterError
        if neither `y` nor `onsets` are provided

        or if `units` is not one of 'frames', 'samples', or 'time'
    """

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    onset_envelope -= onset_envelope.min()

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)

    if normalize:
        # Normalize onset strength function to [0, 1] range
        onset_envelope /= onset_envelope.max()

    # These parameter settings found by large-scale search
    kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
    kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
    kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
    kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("delta", 0.07)

    # Peak pick the onset envelope
    onsets = util.peak_pick(onset_envelope, **kwargs)

    # Optionally backtrack the events
    if backtrack:
        if energy is None:
            energy = onset_envelope

        onsets = onset_backtrack(onsets, energy)

    if units == "frames":
        pass
    elif units == "samples":
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    return onsets


def calc_slice_points(sample: np.ndarray, sr: int = settings.SAMPLE_RATE) -> List[Tuple[int, int]]:
    """Return a list of tuples of slice points in samples."""
    slice_points: List[Tuple[int, int]] = []

    # Calculate the onsets
    onsets = librosa.onset.onset_detect(
        sample,
        sr,
        backtrack=True,
        units="samples",
        # pre_max=7,
        # post_max=7,
        # pre_avg=3,
        # post_avg=3,
        # delta=0.15,
        wait=10,
    )

    # Loop over each onset to the second last one
    for idx, _ in enumerate(onsets[:-1]):
        slice_pair: Tuple[int, int] = (onsets[idx], onsets[idx + 1])
        slice_points.append(slice_pair)

    # Need one slice point from the last onset to the end of the audio
    slice_points.append((onsets[-1], len(sample) - 1))

    return slice_points


def split_oneshots(
    sample: np.ndarray, sr: int = settings.SAMPLE_RATE
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Slice a `sample` into oneshots based on the onset envelope of the waveform.

    Returns a tuple: (slices, slice_points).
    """
    # Get slice points
    slice_points: List[Tuple[int, int]] = calc_slice_points(sample, sr)

    # Slice sample into separate samples by onsets
    slices: List[np.ndarray] = []

    # The first sample starts at the first onset
    for (slice_start, slice_end) in slice_points:
        slice_ = sample[slice_start:slice_end]
        slices.append(slice_)

    return slices, slice_points
