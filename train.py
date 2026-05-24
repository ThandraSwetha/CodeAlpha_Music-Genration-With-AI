from music21 import converter, instrument, note, chord
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
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


sequence_length = 20

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]

    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_in])

    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

print("\nTotal Patterns:", n_patterns)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

network_input = network_input / float(len(pitchnames))

print("\nInput Shape:")

print(network_input.shape)



network_output = to_categorical(network_output)

model = Sequential()

model.add(LSTM(
    256,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))

model.add(Dropout(0.3))

model.add(LSTM(256))

model.add(Dense(128, activation='relu'))

model.add(Dense(network_output.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("\nTraining Started...\n")

model.fit(network_input, network_output, epochs=5, batch_size=64)

model.save("music_model.keras")

print("\nModel Trained Successfully!")