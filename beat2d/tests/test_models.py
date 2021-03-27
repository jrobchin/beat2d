import os

from librosa.core import load

from beat2d import settings
from beat2d import model

# These are known to predict correctly
TEST_SAMPLES = {
    model.CLASSES.KICK: "CZTegYnTPFpnh2dBAQWmtz",
    model.CLASSES.SNARE: "6qmztTaVgp2QGHVGwU8Kwd",
}


def test_beat2dnet():
    """End-to-end test to make sure everything works."""
    m = model.load_beat2dnetv1()
    assert m is not None

    for cls, id in TEST_SAMPLES.items():
        ex_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{id}.wav")
        X, sr = load(ex_path, settings.SAMPLE_RATE)
        assert sr == settings.SAMPLE_RATE

        pred_cls, conf = m.predict(X)
        assert pred_cls == cls
