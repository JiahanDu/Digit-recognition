import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MusicNet dataset and return spectrograms and labels
def load_musicnet_data(data_dir):
    X, y = [], []
    instrument_labels = {}

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            instrument_index = len(instrument_labels)
            instrument_labels[label] = instrument_index

            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                # Load audio and generate spectrogram
                audio, sr = librosa.load(file_path, sr=None)
                stft = np.abs(librosa.stft(audio))
                spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
                spectrogram = librosa.util.fix_length(spectrogram, size=128, axis=1)
                X.append(spectrogram)
                y.append(instrument_index)

    X = np.array(X)[..., np.newaxis]  # Add channel dimension
    y = np.array(y)
    return X, y, instrument_labels

# Load dataset
data_dir = "C:/Users/jiaha/.cache/kagglehub/datasets/imsparsh/musicnet-dataset"
X, y, instrument_labels = load_musicnet_data(data_dir)

# Shuffle and split data into training and testing sets
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Set 80% for training, 20% for testing
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(instrument_labels), activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2%}")
