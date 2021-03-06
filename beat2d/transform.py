from typing import List

import mido
from numpy import np

from beat2d import slicing


def beatbox_to_midi(sample: np.ndarray) -> List[mido.Message]:
    messages: List[mido.Message] = []

    # Slice audio into samples
    slices = slicing.split_oneshots()

    # Convert to time in seconds
    # Classify each sample
    # Create midi message for each sample and time

    return messages
