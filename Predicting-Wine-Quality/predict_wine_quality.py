# Predicting Wine Quality

# Comparison of PCR and PLS Regression Models

# Dataset: Wine Quality Dataset

# Goal: 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

wine = pd.read_csv("WineQT.csv")

df = wine.copy()

df = df.drop(columns=["Id"])

fig, axes = plt.subplots(2, 6, figsize=(18,8))

for i, column in enumerate(df.columns):
    sns.boxplot(x=df[column], ax=axes[i // 6, i % 6])
    axes[i // 6, i % 6].set_title(column)

fig.suptitle("Before", fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


# The function for detecting outliers and replacing them with the mean

def outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    df.loc[outliers, column] = df[column].mean()

    return df

columns_to_clean = ["fixed acidity", "residual sugar", "chlorides",
                    "free sulfur dioxide", "total sulfur dioxide", 
                    "density", "sulphates"]

for column in columns_to_clean:
    df = outliers(df, column)

# -----------------------------------------------------------------------

print("\nBefore:\n", wine.describe().T)
print("\nAfter:\n", df.describe().T)


fig, axes = plt.subplots(2, 6, figsize=(18,8))

for i, column in enumerate(df.columns):
    sns.boxplot(x=df[column], ax=axes[i // 6, i % 6])
    axes[i // 6, i % 6].set_title(column)

fig.suptitle("After", fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
    

corr_matrix = df.corr()
corr_quality = corr_matrix["quality"].sort_values(ascending=False)
print("Correlations with quality")
print(corr_quality)

# We are going to use alcohol, sulphates, citric acid and volatile acidity as undependent variables