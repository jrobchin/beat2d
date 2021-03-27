import os
from typing import List
from unittest import TestCase
from unittest.mock import MagicMock

import librosa

import beat2d


TEST_SEQUENCE = os.path.join(os.path.dirname(__file__), "kskks.wav")


class TestTransform(TestCase):
    def test_beatbox_to_notes(self):
        model_mock: beat2d.model.Beat2dNetBase = MagicMock(beat2d.model.Beat2dNetBase)

        model_mock.predict.side_effect = [
            beat2d.CLASSES.KICK,
            beat2d.CLASSES.SNARE,
            beat2d.CLASSES.KICK,
            beat2d.CLASSES.KICK,
            beat2d.CLASSES.SNARE,
        ]

        sample, _ = librosa.core.load(TEST_SEQUENCE, beat2d.settings.SAMPLE_RATE)

        notes: List[beat2d.NOTES] = beat2d.transform.beatbox_to_notes(model_mock, sample)

        expected_notes: List[beat2d.NOTES] = [
            beat2d.Note(beat2d.NOTES.KICK, start=0.046439909297052155, length=0.44117913832199546),
            beat2d.Note(beat2d.NOTES.SNARE, start=0.4876190476190476, length=0.3947392290249433),
            beat2d.Note(beat2d.NOTES.KICK, start=0.8823582766439909, length=0.3715192743764172),
            beat2d.Note(beat2d.NOTES.KICK, start=1.253877551020408, length=0.255419501133787),
            beat2d.Note(beat2d.NOTES.SNARE, start=1.509297052154195, length=0.3927437641723355),
        ]

        assert len(notes) == len(expected_notes)

        for a, b in zip(notes, expected_notes):
            assert a.type == b.type
            self.assertAlmostEqual(a.start, b.start)
            self.assertAlmostEqual(a.length, b.length)

    def test_serialize_deserialize(self):
        notes: List[beat2d.Note] = [
            beat2d.Note(beat2d.NOTES.KICK, start=0.04643, length=0.4411),
            beat2d.Note(beat2d.NOTES.SNARE, start=0.4876, length=0.3947),
            beat2d.Note(beat2d.NOTES.KICK, start=0.8823, length=0.3715),
            beat2d.Note(beat2d.NOTES.KICK, start=1.2538, length=0.2554),
            beat2d.Note(beat2d.NOTES.SNARE, start=1.5092, length=0.3927),
        ]

        notes_as_json: str = (
            '[{"type": 1, "start": 0.04643, "length": 0.4411}, '
            '{"type": 2, "start": 0.4876, "length": 0.3947}, '
            '{"type": 1, "start": 0.8823, "length": 0.3715}, '
            '{"type": 1, "start": 1.2538, "length": 0.2554}, '
            '{"type": 2, "start": 1.5092, "length": 0.3927}]'
        )

        assert beat2d.utils.notes_to_json(notes) == notes_as_json
        assert beat2d.utils.json_to_notes(notes_as_json) == notes
