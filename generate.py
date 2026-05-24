from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model
import numpy as np
import os
import random


# STEP 1: LOAD NOTES FROM DATASET

notes = []

for file in os.listdir("dataset"):

    if file.endswith(".mid") or file.endswith(".midi"):

        print("Reading:", file)

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


# STEP 2: PREPARE MAPPING

sequence_length = 20

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))


# STEP 3: CREATE INPUT SEQUENCES

network_input = []

for i in range(0, len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_in])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

network_input = network_input / float(len(pitchnames))


# STEP 4: LOAD MODEL

model = load_model("music_model.keras")


# STEP 5: START PREDICTION

start = random.randint(0, len(network_input) - 1)

pattern = network_input[start]

prediction_output = []

print("\nGenerating music...\n")

for note_index in range(100):

    prediction_input = np.reshape(pattern, (1, sequence_length, 1))

    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    # FIXED PATTERN UPDATE (IMPORTANT)
    new_val = index / float(len(pitchnames))
    pattern = np.append(pattern[1:], new_val)
    pattern = np.reshape(pattern, (sequence_length, 1))


# STEP 6: CONVERT TO MIDI

offset = 0
output_notes = []

for pattern in prediction_output:

    # CHORD
    if ('.' in pattern) or pattern.isdigit():

        notes_in_chord = pattern.split('.')
        notes_list = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes_list.append(new_note)

        new_chord = chord.Chord(notes_list)
        new_chord.offset = offset
        output_notes.append(new_chord)

    # SINGLE NOTE
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5


# STEP 7: SAVE FILE

midi_stream = stream.Stream(output_notes)

output_file = "generated_music.mid"
midi_stream.write('midi', fp=output_file)

print("\nMusic Generated Successfully!")
print("Saved file:", output_file)