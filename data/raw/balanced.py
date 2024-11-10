import pandas as pd
from sklearn.utils import resample
import numpy as np


# Load the dataset
df = pd.read_csv('D:/MSIS/CreditCardFraudDetection/data/raw/creditcard.csv')

# Separate majority and minority classes
df_majority = df[df.Class == 0]
df_minority = df[df.Class == 1]

# Determine the number of entries per class for a balanced dataset
entries_per_class = 10000 // 2

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # sample without replacement
                                   n_samples=entries_per_class,     # to match minority class
                                   random_state=42)  # reproducible results

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=entries_per_class,    # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])

# Shuffle the DataFrame
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

# Save to CSV (optional)
df_balanced.to_csv('D:/MSIS/CreditCardFraudDetection/data/raw/balanced_creditcard_10000.csv', index=False)

print(df_balanced['Class'].value_counts())