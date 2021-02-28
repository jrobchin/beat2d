import os
import argparse

from librosa.core import load
import soundfile as sf

from beat2d import settings, split


def main():
    parser = argparse.ArgumentParser(
        description="Split a long recording of oneshots into separate audio files."
    )
    parser.add_argument("label", type=str, help="the label of the sample")
    parser.add_argument("input", type=str, help="path to file containing samples")
    parser.add_argument("outpath", type=str, help="the directory to output slices samples to")
    args = parser.parse_args()

    label: str = args.label
    inputpath: str = args.input
    outpath: str = args.outpath

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    sample = load(inputpath, settings.SAMPLE_RATE)[0]

    oneshots = split.split_oneshots(sample, settings.SAMPLE_RATE)

    # Save each oneshot to a separate audio file
    num_digits = len(str(len(oneshots)))  # Number of digits for the file name

    for idx, oneshot in enumerate(oneshots):
        fname = f"{label}-{idx:0{num_digits}d}.wav"
        fpath = os.path.join(outpath, fname)

        sf.write(fpath, oneshot, settings.SAMPLE_RATE)


if __name__ == "__main__":
    main()
