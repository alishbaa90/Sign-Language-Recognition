import numpy as np

def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(21, 3)
    wrist = coords[0].copy()
    coords = coords - wrist 

    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords = coords / scale

    return coords.flatten().tolist()