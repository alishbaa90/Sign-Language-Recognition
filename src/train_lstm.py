import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

DYNAMIC_WORDS = ["PLEASE", "SORRY", "BYE", "COME"]
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63  # 21 landmarks x 3 coords

# --- Load all sequences ---
X = []
y = []

for idx, word in enumerate(DYNAMIC_WORDS):
    path = f"../data/lstm/{word}.npy"
    sequences = np.load(path)  # shape: (num_sequences, 30, 63)
    X.append(sequences)
    y.extend([idx] * len(sequences))

X = np.concatenate(X, axis=0)  # shape: (total_sequences, 30, 63)
y = np.array(y)
y_categorical = to_categorical(y, num_classes=len(DYNAMIC_WORDS))

print(f"Dataset shape: X={X.shape}, y={y_categorical.shape}")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# --- Build the LSTM model ---
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dropout(0.3),
    LSTM(32, return_sequences=False, activation='tanh'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(DYNAMIC_WORDS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop]
)

# --- Evaluate ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# --- Save model + label mapping ---
os.makedirs("../models/lstm", exist_ok=True)
model.save("../models/lstm/dynamic_gesture_model.keras")

with open("../models/lstm/label_map.pkl", "wb") as f:
    pickle.dump(DYNAMIC_WORDS, f)

print("Model saved to models/lstm/dynamic_gesture_model.keras")