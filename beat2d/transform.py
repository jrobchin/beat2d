from enum import Enum
from typing import List, Dict, Union
from dataclasses import dataclass

import librosa
import numpy as np

import beat2d


class NOTES(Enum):
    KICK = 1
    SNARE = 2
    HAT = 3


NOTE_MAP: Dict[beat2d.CLASSES, int] = {
    beat2d.CLASSES.KICK: NOTES.KICK,
    beat2d.CLASSES.SNARE: NOTES.SNARE,
    beat2d.CLASSES.HAT: NOTES.HAT,
}


@dataclass
class Note:
    type: NOTES
    start: float
    length: float

    def __init__(self, type: Union[NOTES, int], start: float, length: float):
        if isinstance(type, int):
            self.type = NOTES(type)
        elif isinstance(type, NOTES):
            self.type = type
        else:
            raise TypeError("type must be an int or NOTES enum")

        self.start = start
        self.length = length


def beatbox_to_notes(
    classifier: beat2d.model.Beat2dNetBase,
    sample: np.ndarray,
    sr: int = beat2d.settings.SAMPLE_RATE,
) -> List[Note]:
    notes: List[Note] = []

    slices, slice_pts = beat2d.slicing.split_oneshots(sample, sr)

    for slc, slc_pts in zip(slices, slice_pts):
        start_time = librosa.core.samples_to_time(slc_pts[0], sr)
        end_time = librosa.core.samples_to_time(slc_pts[1], sr)

        length: float = end_time - start_time

        pred: beat2d.CLASSES
        conf: float
        pred, conf = classifier.predict(slc)

        if pred == beat2d.CLASSES.NONE:
            continue

        note: Note = Note(NOTE_MAP[pred], start_time, length)
        notes.append(note)

    return notes


def notes_to_audio(
    notes: List[Note], samples: Dict[NOTES, np.ndarray], sr: int = beat2d.settings.SAMPLE_RATE
) -> np.ndarray:
    """Render a list of notes to audio.

    Args:
        notes: A list of `Note`s to turn into audio.
        samples: A dict mapping the NOTES enum to waveforms which contain the replacement samples.

    Returns:
        Waveform of the resulting audio.
    """

    # Check that we don't have an empty list
    if len(notes) == 0:
        raise ValueError("notes cannot be empty")

    # Calculate the length of the resulting audio. We need this for the length of the array.
    # Add the last note's starting point to it's length
    audio_length_seconds: float = notes[-1].start + notes[-1].length
    audio_length_samples: int = librosa.core.time_to_samples(audio_length_seconds, sr)

    # Create an array of silence
    out: np.ndarray = np.zeros((audio_length_samples,), dtype=np.float32)

    # For each note, add the corresponding sample
    for note in notes:
        if note.type not in samples.keys():
            raise ValueError(f"Note type: {note.type} was not found in `samples`.")

        sample: np.ndarray = samples[note.type]

        start: int = librosa.core.time_to_samples(note.start, sr)
        length: int = librosa.core.time_to_samples(note.length, sr)
        length = min(length, sample.shape[0])
        end: int = start + length

        # Fill in the output with the contents of the sample
        out[start:end] = sample[:length]

    return out


def beatbox_to_audio(
    classifier: beat2d.model.Beat2dNetBase,
    beatbox_waveform: np.ndarray,
    samples: Dict[NOTES, np.ndarray],
    sr: int = beat2d.settings.SAMPLE_RATE,
) -> np.ndarray:
    notes: List[Note] = beatbox_to_notes(classifier, beatbox_waveform, sr)
    output: np.ndarray = notes_to_audio(notes, samples, sr)
    return output
