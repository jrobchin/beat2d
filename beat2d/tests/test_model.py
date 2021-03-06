import os

from librosa.core import load

from beat2d import settings
from beat2d.model import load_model, CLASSES
from beat2d.data import get_dataset

# These are known to predict correctly
TEST_SAMPLES = {
    CLASSES.NONE: "Q7gFWKMVBsFBQ47WW5qhFA",
    CLASSES.KICK: "CZTegYnTPFpnh2dBAQWmtz",
    CLASSES.SNARE: "6qmztTaVgp2QGHVGwU8Kwd",
}


def test_predicts_correct_class():
    model = load_model()
    assert model is not None

    dataset = get_dataset()

    for cls, id in TEST_SAMPLES.items():
        ex = dataset[dataset.id == id].iloc[0]
        assert ex.label == cls.name.lower()

        # TODO: change dataset csv so that path is relative to top-level directory
        ex_path = os.path.join(settings.DATA_DIR, "oneshots", f"{id}.wav")
        X, sr = load(ex_path, settings.SAMPLE_RATE)
        assert sr == settings.SAMPLE_RATE

        pred = model.predict(X)
        assert pred == cls
