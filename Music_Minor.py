# READ WHOLE FOLDER FOR TRAINING..

import os
import numpy as np
import tensorflow as tf
from music21 import converter, note, chord
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam



# Step 0 : Set memory growth for GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Step 1: Data Preprocessing
kern_files_directory = "kern file" # change the dir where .krn files are located.
kern_files = os.listdir(kern_files_directory)

# Parse the KRISTALISER files and extract musical features
notes = []
for file in kern_files:
    midi = converter.parse(os.path.join(kern_files_directory, file))
    notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# Convert notes to numeric representation
note_to_int = dict((note, number) for number, note in enumerate(sorted(set(notes))))

# Define the sequence length for input data
sequence_length = 100  # Define the length of input sequences

# Prepare input sequences and corresponding output labels
network_input = []
network_output = []
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

# Reshape and normalize the input data
X = np.reshape(network_input, (len(network_input), sequence_length, 1))
X = X / float(len(note_to_int))
y = tf.keras.utils.to_categorical(network_output)

# Step 2: Model Construction (GNN)
# Define your GNN architecture
input_shape = (sequence_length, 1)
inputs = Input(shape=input_shape)
x = LSTM(256, return_sequences=True)(inputs)
x = LSTM(256)(x)
outputs = Dense(len(note_to_int), activation='softmax')(x)

# Compile the model
model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(lr=0.001)  # Adjust learning rate if needed
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Step 3: Training
# Train your GNN model with X and y
model.fit(X, y, epochs=100, batch_size=30)

# Step 4: Save the Model
# Save your trained GNN model for future use
model.save("flow/model")  # Change the location of model where to save it.

print("Model trained and saved successfully.")

# Step 4: Melody Generation
# Generate a seed sequence
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]

int_to_note = dict((number, note) for number, note in enumerate(notes))
# Generate notes
prediction_output = []
for note_index in range(500):  # Generate 500 notes
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(note_to_int))

    prediction = model.predict(prediction_input, verbose=0)

    # Sample the index of the next note based on the prediction
    index = np.argmax(prediction)

    result = int_to_note[index]

    # Append the predicted note to the output
    prediction_output.append(result)

    # Update the pattern with the new note
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# Step 5: Save the Generated Melody
from music21 import stream

# Convert the output to a stream of Note objects
output_notes = []
for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():  # If pattern is a chord
        notes_in_chord = pattern.split('.')
        notes = [note.Note(int_to_note[int(note_in_chord)]) for note_in_chord in notes_in_chord]
        chord_note = chord.Chord(notes)
        chord_note.duration.quarterLength = 0.5  # Adjust duration if needed
        output_notes.append(chord_note)
    else:  # If pattern is a single note
        new_note = note.Note(pattern)
        new_note.duration.quarterLength = 0.5  # Adjust duration if needed
        output_notes.append(new_note)

# Create a stream object and add notes to it
midi_stream = stream.Stream(output_notes)

# Save the stream as a MIDI file
output_file_path = "Outputtest.mid" # change the dir where output you want to save it.
midi_stream.write('midi', fp=output_file_path)

print("Generated melody saved successfully.")
