import torchaudio

torchaudio.set_audio_backend("sox_io")

from beat2d.model import CLASSES
from beat2d.transform import NOTES, Note, beatbox_to_audio

from beat2d import settings, model, utils, slicing
