import pandas as pd

df = pd.read_csv("data/raw/atlas-higgs-challenge-2014-v2.csv.gz")

print("shape:", df.shape)
print()
print("labels (s = signal, b = background):")
print(df["Label"].value_counts())
print()

feature_cols = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
print("number of features:", len(feature_cols))
print()

print("fraction of -999 (undefined) values per feature:")
missing = (df[feature_cols] == -999.0).mean().sort_values(ascending=False)
print(missing[missing > 0])
print()

print("jet counts:")
print(df["PRI_jet_num"].value_counts().sort_index())
print()

print("KaggleSet values:")
print(df["KaggleSet"].value_counts())