from enum import Enum

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from beat2d import settings

torchaudio.set_audio_backend("sox_io")

MAX_CLIP_LENGTH = librosa.core.time_to_samples(0.2)


class CLASSES(Enum):
    NONE = 0
    KICK = 1
    SNARE = 2


class Model(nn.Module):
    num_fully_connected_features = 16 * 16 * 4

    def __init__(self, sr: int = settings.SAMPLE_RATE):
        super(Model, self).__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=70, normalized=True)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(self.num_fully_connected_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.melspec(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_fully_connected_features)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.33)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        # Normalize the audio
        s_norm = librosa.util.normalize(waveform)

        # Trim silence
        s_trim = librosa.effects.trim(s_norm, frame_length=256, hop_length=64, top_db=25)[0]

        # Clip or pad to MAX_CLIP_LENGTH
        s_dest = np.zeros(MAX_CLIP_LENGTH, dtype=np.float32)
        clip_length = min(MAX_CLIP_LENGTH, s_trim.shape[0])
        s_dest[:clip_length] = s_trim[:clip_length]

        return torch.from_numpy(s_dest).view(1, 4410)

    def predict(self, waveform: np.ndarray) -> CLASSES:
        x = self.preprocess(waveform).unsqueeze(0)
        x = self.forward(x)
        return CLASSES(int(F.softmax(x, dim=1).argmax()))


def load_model(path: str = settings.MODEL_PATH):
    m = Model()
    m.load_state_dict(torch.load(path))
    m.eval()

    return m
