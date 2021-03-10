import json
from typing import List
from dataclasses import asdict

from beat2d import Note


def notes_to_dicts(notes: List[Note]):
    ret: List[dict] = []

    for note in notes:
        d: dict = asdict(note)
        d["type"] = d["type"].value

        ret.append(d)

    return ret


def notes_to_json(notes: List[Note]):
    return json.dumps(notes_to_dicts(notes))


def json_to_notes(s: str):
    note_dicts: List[dict] = json.loads(s)

    notes: List[Note] = []

    for n in note_dicts:
        note: Note = Note(**n)
        notes.append(note)

    return notes
