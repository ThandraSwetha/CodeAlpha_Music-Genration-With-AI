# AI Music Generator

An advanced AI-powered music generation system built with deep learning that generates unique MIDI music files based on patterns learned from training data.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [How It Works](#how-it-works)

---

## Overview

This project implements a machine learning-based music generation system that:
- **Preprocesses MIDI files** into numerical sequences
- **Trains a neural network** to learn musical patterns
- **Generates new music** based on learned patterns
- **Saves output as MIDI files** for playback in music software

The system uses Keras/TensorFlow for building and training the deep learning model.

---

## Project Structure

```
AI_Music_Generator/
├── dataset/                          # Training data folder
│   ├── maa-tujhe-salaam-easy-solo-sheet_MattE_1.mid
│   ├── National Anthem - India.mid
│   ├── saare-jahan-se-achcha.mid
│   └── vande-mataram.mid
├── preprocess.py                     # Data preprocessing script
├── train.py                          # Model training script
├── generate.py                       # Music generation script
├── test.py                           # Testing/evaluation script
├── music_model.keras                 # Trained model file
├── generated_music.mid               # Output MIDI file
└── README.md                         # Documentation
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/ThandraSwetha/CodeAlpha_Music-Genration-With-AI.git
cd CodeAlpha_Music-Genration-With-AI
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. **Install required dependencies:**
```bash
pip install tensorflow keras music21 numpy
```

---

## File Descriptions

### 1. **preprocess.py** - Data Preprocessing

**Purpose:** Converts MIDI files into numerical sequences that can be fed into the neural network.

**Line-by-line execution:**

```python
from music21 import converter, instrument, note, chord
import numpy as np
import pickle
import os
```
- Imports music processing library (`music21`), numerical computing (`numpy`), file handling (`pickle`), and OS utilities

```python
notes = []
```
- Initializes an empty list to store all musical notes and chords

```python
for file in os.listdir("dataset"):
    if file.endswith(".mid"):
        try:
            midi = converter.parse(f"dataset/{file}")
```
- Loops through all MIDI files in the dataset folder
- Uses `converter.parse()` to load each MIDI file

```python
            notes_to_parse = midi.flatten().notes
```
- Extracts all notes from the MIDI file structure
- `.flatten()` removes nested structures for easier access

```python
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.pitches))
```
- For each note: stores its pitch value as a string
- For each chord: stores all pitches separated by dots (e.g., "C4.E4.G4")

```python
        except Exception as e:
            print(f"Error processing {file}: {e}")
```
- Catches and reports any errors during file processing

```python
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)
```
- Saves the list of notes to a pickle file for later use in training

---

### 2. **train.py** - Model Training

**Purpose:** Builds and trains a neural network to learn musical patterns from preprocessed data.

**Line-by-line execution:**

```python
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
```
- Imports necessary libraries for neural network construction and training

```python
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)
```
- Loads the preprocessed notes from the pickle file

```python
unique_notes = sorted(set(notes))
note_to_num = {note: num for num, note in enumerate(unique_notes)}
num_to_note = {num: note for note, num in note_to_num.items()}
```
- Creates unique note list and two dictionaries for encoding/decoding:
  - `note_to_num`: converts notes (strings) to numbers
  - `num_to_note`: converts numbers back to note strings

```python
sequence_length = 100
X, y = [], []
```
- Sets sequence length to 100 notes
- `X` will store input sequences, `y` will store target notes

```python
for i in range(len(notes) - sequence_length):
    X.append([note_to_num[note] for note in notes[i:i+sequence_length]])
    y.append(note_to_num[notes[i+sequence_length]])
```
- Creates training sequences:
  - Takes 100 consecutive notes as input (X)
  - Takes the next note as the target output (y)
  - Repeats for all positions in the dataset

```python
X = np.array(X) / len(unique_notes)
y = keras.utils.to_categorical(np.array(y), num_classes=len(unique_notes))
```
- Normalizes X by dividing by the number of unique notes
- Converts y to one-hot encoding (e.g., [0,0,1,0...] for classification)

```python
model = Sequential([
    LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(512),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(len(unique_notes), activation='softmax')
])
```
- Builds the neural network:
  - **LSTM layers**: Long Short-Term Memory layers capture patterns in sequential data
  - **Dropout(0.2)**: Randomly deactivates 20% of neurons to prevent overfitting
  - **Dense(256)**: Fully connected layer with 256 neurons and ReLU activation
  - **Dense(output)**: Output layer with softmax for classification

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
- Configures the model:
  - `loss`: Measures prediction error (categorical_crossentropy for multi-class)
  - `optimizer`: Adam optimizer adjusts weights to minimize loss
  - `metrics`: Tracks accuracy during training

```python
model.fit(X, y, epochs=50, batch_size=64)
```
- **Trains the model** for 50 epochs:
  - Each epoch processes the entire dataset
  - `batch_size=64`: Processes 64 samples at a time
  - Weights are updated after each batch

```python
model.save("music_model.keras")
```
- Saves the trained model to disk for later use in generation

```python
with open("mappings.pkl", "wb") as f:
    pickle.dump({"note_to_num": note_to_num, "num_to_note": num_to_note}, f)
```
- Saves the note-to-number mappings for use during generation

---

### 3. **generate.py** - Music Generation

**Purpose:** Uses the trained model to generate new music based on learned patterns.

**Line-by-line execution:**

```python
import pickle
import numpy as np
from tensorflow import keras
from music21 import note, chord, stream, instrument
```
- Imports libraries for model loading, note creation, and MIDI generation

```python
model = keras.models.load_model("music_model.keras")
```
- Loads the pre-trained model from the saved file

```python
with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
    note_to_num = mappings["note_to_num"]
    num_to_note = mappings["num_to_note"]
```
- Loads the note mappings created during training

```python
seed_sequence = [note_to_num[note] for note in [random_notes...]]
seed_sequence = np.array(seed_sequence) / len(unique_notes)
```
- Creates a starting sequence (seed) with random notes from training data
- Normalizes the seed sequence to match training normalization

```python
for _ in range(500):
    prediction = model.predict(seed_sequence.reshape(1, 100, 1))
    note_index = np.argmax(prediction)
    generated_sequence.append(num_to_note[note_index])
```
- **Generates 500 new notes**:
  - `model.predict()`: Predicts the next note based on current sequence
  - `np.argmax()`: Picks the note with highest probability
  - Adds the predicted note to the output sequence

```python
seed_sequence = np.append(seed_sequence[1:], note_index/len(unique_notes))
```
- **Slides the sequence window**:
  - Removes the first note from seed_sequence
  - Adds the new predicted note at the end
  - This maintains a 100-note input window for continuous generation

```python
midi_stream = stream.Score()
midi_stream.append(instrument.Piano())
```
- Creates a new MIDI score and adds a piano instrument

```python
for note_str in generated_sequence:
    if '.' in note_str:
        pitches = note_str.split('.')
        chord_obj = chord.Chord(pitches)
        chord_obj.quarterLength = 0.5
        midi_stream.append(chord_obj)
    else:
        note_obj = note.Note(note_str)
        note_obj.quarterLength = 0.5
        midi_stream.append(note_obj)
```
- Converts string notes back to music21 objects:
  - If note contains '.': it's a chord, create chord object
  - Otherwise: create a single note object
  - `quarterLength = 0.5`: Sets duration (0.5 = half beat)

```python
midi_stream.write('midi', fp='generated_music.mid')
```
- Writes the complete MIDI sequence to an output file

---

### 4. **test.py** - Testing & Evaluation

**Purpose:** Validates model performance and tests generation quality.

**Line-by-line execution:**

```python
model = keras.models.load_model("music_model.keras")
```
- Loads the trained model

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
```
- Evaluates model on test data
- Prints accuracy percentage

```python
# Generate sample music and listen
output_midi = generate_music(model, mappings, num_notes=200)
```
- Generates a sample MIDI file with custom parameters
- Can be played back in DAWs or media players

---

## Usage

### Step 1: Preprocess Data
```bash
python preprocess.py
```
**Output:** Creates `notes.pkl` containing all extracted notes from MIDI files

### Step 2: Train the Model
```bash
python train.py
```
**Output:** 
- Saves trained model as `music_model.keras`
- Saves note mappings as `mappings.pkl`
- Displays training progress and accuracy

### Step 3: Generate Music
```bash
python generate.py
```
**Output:** Creates `generated_music.mid` - a playable MIDI file with AI-generated music

### Step 4: Test the Model (Optional)
```bash
python test.py
```
**Output:** Displays model accuracy and generates test samples

---

## How It Works

### The Process Flow:

```
MIDI Dataset
    ↓
[PREPROCESS] → Extract notes/chords
    ↓
Numerical Sequences (encoded notes)
    ↓
[TRAIN] → Feed to LSTM Neural Network
    ↓
Trained Model (learns patterns)
    ↓
[GENERATE] → Predict next notes iteratively
    ↓
Generated Sequence
    ↓
[CONVERT] → Back to notes/chords
    ↓
Generated MIDI File
```

### Key Technologies:

1. **Music21 Library**: Parses and creates MIDI files
2. **LSTM Networks**: Long Short-Term Memory units preserve long-term dependencies in music sequences
3. **Keras/TensorFlow**: Deep learning framework for model training
4. **NumPy**: Numerical computing for array operations

---

## Parameters & Customization

### In `train.py`:
- `epochs`: Increase to 100+ for better results (slower training)
- `batch_size`: Adjust based on system RAM (default 64)
- LSTM units: Increase 256/512 for more complex patterns

### In `generate.py`:
- `num_generations`: Control how many notes to generate (default 500)
- `seed_length`: Adjust starting sequence length

### In `preprocess.py`:
- `sequence_length`: Change from 100 to capture longer/shorter patterns

---

## Example Output

The model generates MIDI files that contain:
- Individual notes with realistic pitch variations
- Chords from the training data
- Rhythmic patterns learned from input music
- Smooth transitions between notes

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'music21'` | Run `pip install music21` |
| `CUDA out of memory` | Reduce `batch_size` in train.py |
| Generated music sounds random | Increase training epochs or dataset size |
| Model file not found | Run train.py before generate.py |

---

## Future Enhancements

- Add tempo and time signature variations
- Implement genre-specific models
- Add user control over musical style
- Web interface for real-time generation
- Support for polyphonic music generation

---

## License

This project is part of CodeAlpha AI internship program.

---

## Contact

For questions or suggestions, contact the development team.

---

**Happy Music Generation! 🎵**
