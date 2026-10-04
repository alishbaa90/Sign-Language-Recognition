import pandas as pd
import glob
from utils import normalize_landmarks

for file in glob.glob("../data/*.csv"):
    df = pd.read_csv(file, header=None)
    normalized_rows = df.apply(lambda row: normalize_landmarks(row.tolist()), axis=1)
    new_df = pd.DataFrame(normalized_rows.tolist())
    new_df.to_csv(file, header=False, index=False)
    print(f"Normalized: {file}")