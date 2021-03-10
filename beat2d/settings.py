import os

import torch


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DATASET_PATH = os.path.join(DATA_DIR, "oneshots", "_labels.csv")
BEATNETV1_MODEL_PATH = os.path.join(BASE_DIR, "basenetv1.pth")
BEATNETV2_MODEL_PATH = os.path.join(BASE_DIR, "basenetv2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_DIR = os.path.join(os.path.dirname(BASE_DIR), "samples")

SAMPLE_RATE = 22050

if __name__ == "__main__":
    print("BASE_DIR", BASE_DIR)
    print("DATA_DIR", DATA_DIR)
    print("DATASET_PATH", DATASET_PATH)
    print("BEATNETV1_MODEL_PATH", BEATNETV1_MODEL_PATH)
    print("BEATNETV2_MODEL_PATH", BEATNETV2_MODEL_PATH)
    print("DEVICE", DEVICE)
