import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DATASET_PATH = os.path.join(DATA_DIR, "oneshots", "_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

SAMPLE_RATE = 22050

if __name__ == "__main__":
    print("BASE_DIR", BASE_DIR)
    print("DATA_DIR", DATA_DIR)