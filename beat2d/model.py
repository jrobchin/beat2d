from enum import Enum

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

torchaudio.set_audio_backend("sox_io")

from beat2d import settings


class CLASSES(Enum):
    NONE = -1
    KICK = 0
    SNARE = 1
    HAT = 2


class Beat2dNetBase(nn.Module):
    def __init__(self, device: str, max_clip_length: int, sr: int = settings.SAMPLE_RATE):
        super().__init__()
        self.device = device
        self.max_clip_length = max_clip_length

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        # Normalize the audio
        s_norm = librosa.util.normalize(waveform)

        # Trim silence
        s_trim = librosa.effects.trim(s_norm, frame_length=256, hop_length=64, top_db=25)[0]

        # Clip or pad to self.max_clip_length
        s_dest = np.zeros(self.max_clip_length, dtype=np.float32)
        clip_length = min(self.max_clip_length, s_trim.shape[0])
        s_dest[:clip_length] = s_trim[:clip_length]

        return torch.from_numpy(s_dest).to(self.device).view(1, self.max_clip_length)

    def predict(self, waveform: np.ndarray, thresh: float = 0.7) -> CLASSES:
        x = self.preprocess(waveform).unsqueeze(0)
        x = self.forward(x)

        preds: torch.Tensor = F.softmax(x[0])
        class_pred: int = int(preds.argmax())
        if preds[class_pred] > thresh:
            return CLASSES(class_pred), preds[class_pred]
        else:
            return CLASSES.NONE, preds[class_pred]


class Beat2dNetV1(Beat2dNetBase):
    max_clip_length: int = 8820
    num_fc_features: int = 16 * 16 * 9

    def __init__(self, device: str, sr: int = settings.SAMPLE_RATE):
        super().__init__(device, self.max_clip_length, sr)
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=70, normalized=True)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(self.num_fc_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.melspec(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_fc_features)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def load_beat2dnetv1(
    device: str = settings.DEVICE, path: str = settings.BEAT2DNETV1_MODEL_PATH, sr: int = settings.SAMPLE_RATE
):
    m = Beat2dNetV1(device, sr).to(device)
    m.load_state_dict(torch.load(path))
    m.eval()
    return m
