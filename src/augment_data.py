import pandas as pd
import numpy as np
import glob
import os

def augment_landmarks(landmarks, rotation_deg=10, noise_std=0.02, scale_range=0.1):
    coords = np.array(landmarks).reshape(21, 3)

    # Random rotation around z-axis (in the hand's own plane)
    theta = np.radians(np.random.uniform(-rotation_deg, rotation_deg))
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords = coords @ rot_matrix.T

    # Random scaling
    scale = 1 + np.random.uniform(-scale_range, scale_range)
    coords = coords * scale

    # Random Gaussian noise
    coords = coords + np.random.normal(0, noise_std, coords.shape)

    return coords.flatten().tolist()

def augment_dataset(augment_factor=3):
    for file in glob.glob("../data/*.csv"):
        df = pd.read_csv(file, header=None)
        original_rows = df.values.tolist()

        augmented_rows = []
        for row in original_rows:
            for _ in range(augment_factor):
                augmented_rows.append(augment_landmarks(row))

        all_rows = original_rows + augmented_rows
        new_df = pd.DataFrame(all_rows)
        new_df.to_csv(file, header=False, index=False)
        print(f"{os.path.basename(file)}: {len(original_rows)} -> {len(all_rows)} samples")

if __name__ == "__main__":
    augment_dataset(augment_factor=3)