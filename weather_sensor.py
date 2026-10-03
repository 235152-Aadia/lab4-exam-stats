import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

# Create the 15-reading weather sensor dataset
data = {
    "Hour": range(1, 16),
    "Temperature": [
        22, 23, np.nan, np.nan, np.nan,
        25, 26, 58, 24, 23,
        22, 21, 23, 24, 25
    ]
}

df = pd.DataFrame(data)

# Step 1: Display statistics BEFORE cleaning
print("Stats BEFORE cleaning:")

print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())
print("Skewness:", stats.skew(df["Temperature"].dropna()))
print("Kurtosis:", stats.kurtosis(df["Temperature"].dropna()))

# Save the raw temperature values
df["Temp_raw"] = df["Temperature"]

# Step 2: Fill missing values using linear interpolation
df["Temperature"] = df["Temperature"].interpolate(method="linear")

print("\nAfter interpolation, missing values:",
      df["Temperature"].isnull().sum())

# Step 3: Detect outlier using Z-score threshold of 2
z_scores = zscore(df["Temperature"])

# Find values where absolute Z-score is greater than 2
outlier_mask = np.abs(z_scores) > 2

# Display detected outlier
outlier_hours = df.loc[outlier_mask, "Hour"].tolist()

for hour in outlier_hours:
    value = df.loc[df["Hour"] == hour, "Temperature"].iloc[0]
    print(
        "Z-score(>2) outlier detected at Hour:",
        hour,
        "(value:", value, ")"
    )

# Step 4: Replace the flagged outlier with the column median
median_temperature = df["Temperature"].median()

df.loc[outlier_mask, "Temperature"] = median_temperature

print("Replaced with column median:", median_temperature)

# Step 5: Apply Z-score standardization after cleaning
df["Temp_Zscore"] = zscore(df["Temperature"])

# Step 6: Create before/after comparison table
comparison = df[[
    "Hour",
    "Temp_raw",
    "Temperature",
    "Temp_Zscore"
]].copy()

comparison.columns = [
    "Hour",
    "Temp_raw",
    "Temp_clean",
    "Temp_Zscore"
]

print("\nBefore/After comparison:")
print(comparison.to_string(index=False))

# Step 7: Statistics AFTER cleaning
print("\nStats AFTER cleaning:")
print("Mean:", df["Temperature"].mean())
print("Std:", df["Temperature"].std())

