import warnings
warnings.filterwarnings("ignore")
import os
import argparse

import librosa
import soundfile
import numpy as np

import beat2d


DEFAULT_KICK_SAMPLE = os.path.join(beat2d.settings.SAMPLE_DIR, "kick1.wav")
DEFAULT_SNARE_SAMPLE = os.path.join(beat2d.settings.SAMPLE_DIR, "snare1.wav")
DEFAULT_HAT_SAMPLE = os.path.join(beat2d.settings.SAMPLE_DIR, "hat1.wav")


def main():
    parser = argparse.ArgumentParser(description="Transform audio of beatboxing to drums.")
    parser.add_argument("input", type=str, help="path to file containing beatboxing")
    parser.add_argument("outpath", type=str, help="path to save resulting audio")
    args = parser.parse_args()

    input_path: str = args.input
    output_path: str = args.outpath

    print("Loading samples...")
    samples: dict = {
        beat2d.NOTES.KICK: librosa.core.load(DEFAULT_KICK_SAMPLE, beat2d.settings.SAMPLE_RATE)[0],
        beat2d.NOTES.SNARE: librosa.core.load(DEFAULT_SNARE_SAMPLE, beat2d.settings.SAMPLE_RATE)[0],
        beat2d.NOTES.HAT: librosa.core.load(DEFAULT_HAT_SAMPLE, beat2d.settings.SAMPLE_RATE)[0],
    }
    print("Samples loaded.")

    print("Loading model...")
    classifier: beat2d.model.BeatNetV2 = beat2d.model.load_beat2dnetv1()
    print("Model loaded into memory...")

    print("Reading input audio...")
    input_waveform: np.ndarray = librosa.core.load(input_path, beat2d.settings.SAMPLE_RATE)[0]
    print("Loaded input audio.")

    print("Starting beatbox to audio process...")
    output_waveform: np.ndarray = beat2d.beatbox_to_audio(
        classifier, input_waveform, samples, beat2d.settings.SAMPLE_RATE
    )
    print("Completed beatbox to audio.")

    print(f"Writing drums audio to file {output_path}")
    soundfile.write(output_path, output_waveform, beat2d.settings.SAMPLE_RATE)
    print("Done")


if __name__ == "__main__":
    main()
