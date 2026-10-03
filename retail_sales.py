import pandas as pd
import numpy as np
from scipy.stats import zscore

# Create the 10-day retail sales dataset
data = {
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}

df = pd.DataFrame(data)

# Step 1: Check missing values
print("Missing values in Revenue:", df["Revenue"].isnull().sum())

# Step 2: Fill missing values with median
median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nAfter median fill:")
print(df)

# Step 3: Min-Max normalization
df["Revenue_MinMax"] = (
    (df["Revenue"] - df["Revenue"].min()) /
    (df["Revenue"].max() - df["Revenue"].min())
)

# Step 4: Z-score standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])

# Step 5: IQR outlier detection
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create Outlier Boolean column
df["Outlier"] = (
    (df["Revenue"] < lower_bound) |
    (df["Revenue"] > upper_bound)
)

# Display detected outlier days
outlier_days = df.loc[df["Outlier"], "Day"].tolist()

print("\nIQR outliers detected on day(s):", outlier_days)

# Step 6: Display final table
print("\nFinal Table:")
print(df.to_string(index=False))

