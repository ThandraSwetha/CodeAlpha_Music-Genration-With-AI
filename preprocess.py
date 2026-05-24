from music21 import converter, instrument, note, chord
import os

notes = []

for file in os.listdir("dataset"):

    if file.endswith(".mid") or file.endswith(".midi"):

        print("Reading File:", file)

        midi = converter.parse(f"dataset/{file}")

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()

        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):

                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):

                notes.append('.'.join(str(n) for n in element.normalOrder))

print("\nTotal Notes Extracted:", len(notes))

print("\nFirst 50 Notes:")

print(notes[:50])