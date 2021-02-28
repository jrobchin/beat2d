import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

SAMPLE_RATE = 22050

if __name__ == "__main__":
    print("BASE_DIR", BASE_DIR)
    print("DATA_DIR", DATA_DIR)