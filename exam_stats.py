import pandas as pd
import numpy as np
from scipy import stats

# Exam scores of 10 students
scores = pd.Series([72, 65, 88, np.nan, 54, 91, 76, np.nan, 83, 69])

# Descriptive statistics
print("Descriptive Statistics:")
print(scores.describe())

# Skewness and Kurtosis
clean_scores = scores.dropna()

print("\nSkewness:", stats.skew(clean_scores))
print("Kurtosis:", stats.kurtosis(clean_scores))

# Count missing values
print("\nMissing values:", scores.isnull().sum())

# Fill missing values with mean
filled_scores = scores.fillna(scores.mean())

print("\nFilled scores:")
print(filled_scores)