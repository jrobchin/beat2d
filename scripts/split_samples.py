import os
import argparse
from typing import List
import csv

from librosa.core import load
import numpy as np
import soundfile as sf
import shortuuid

from beat2d import settings, slicing


def main():
    parser = argparse.ArgumentParser(
        description="Split a long recording of oneshots into separate audio files."
    )
    parser.add_argument(
        "inputs", type=str, help="path to file containing samples, comma separated for multiple"
    )
    parser.add_argument("outpath", type=str, help="the directory to output slices samples to")
    parser.add_argument("indexout", type=str, help="path to save the csv file that keeps track of oneshots")
    args = parser.parse_args()

    inputs: str = args.inputs
    outpath: str = args.outpath
    indexout: str = args.indexout

    inputpaths: List[str] = []
    if "," in inputs:
        inputpaths = inputs.split(",")
    else:
        inputpaths = [inputs]

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    if os.path.exists(indexout):
        raise ValueError(
            f"The file {indexout} already exists. " "Please choose a different location for the index file."
        )

    if not os.path.isdir(os.path.dirname(indexout)):
        os.makedirs(os.path.dirname(indexout))

    ids_used = set()
    with open(indexout, "w") as f:
        cols: List[str] = ["id", "oneshot", "source", "start", "end"]
        writer = csv.DictWriter(f, fieldnames=cols)

        writer.writeheader()

        for inputpath in inputpaths:
            recording: np.ndarray = load(inputpath, settings.SAMPLE_RATE)[0]
            oneshots, slice_points = slicing.split_oneshots(recording, settings.SAMPLE_RATE)

            for oneshot, slice_pts in zip(oneshots, slice_points):
                file_id = shortuuid.uuid()
                while file_id in ids_used:
                    file_id = shortuuid.uuid()
                ids_used.add(file_id)

                fname = f"{file_id}.wav"
                fpath = os.path.join(outpath, fname)

                sf.write(fpath, oneshot, settings.SAMPLE_RATE)

                writer.writerow(
                    {
                        "id": file_id,
                        "oneshot": fpath,
                        "source": inputpath,
                        "start": slice_pts[0],
                        "end": slice_pts[1],
                    }
                )


if __name__ == "__main__":
    main()
