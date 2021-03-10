import os

from librosa.core import load

from beat2d import settings
from beat2d import model

# These are known to predict correctly
TEST_SAMPLES = {
    model.CLASSES.NONE: "Q7gFWKMVBsFBQ47WW5qhFA",
    model.CLASSES.KICK: "CZTegYnTPFpnh2dBAQWmtz",
    model.CLASSES.SNARE: "6qmztTaVgp2QGHVGwU8Kwd",
}


def test_beatnetv1():
    """End-to-end test to make sure everything works."""
    m = model.load_beatnetv1()
    assert m is not None

    for cls, id in TEST_SAMPLES.items():
        ex_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{id}.wav")
        X, sr = load(ex_path, settings.SAMPLE_RATE)
        assert sr == settings.SAMPLE_RATE

        pred = m.predict(X)
        assert pred == cls


def test_beatnetv2():
    """End-to-end test to make sure everything works."""
    m = model.load_beatnetv2()
    assert m is not None

    for cls, id in TEST_SAMPLES.items():
        ex_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{id}.wav")
        X, sr = load(ex_path, settings.SAMPLE_RATE)
        assert sr == settings.SAMPLE_RATE

        pred = m.predict(X)
        assert pred == cls
