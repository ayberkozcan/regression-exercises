# Predicting Crime Rate in Boston

# Simple and Multiple Linear Regression

# Dataset: Boston House Prices-Advanced Regression Techniques

# Goal: Predicting the crime rate based on property-tax rate and accessibility to radial highways with simple linear regression or multiple linear regression

# There are no missing values in the dataset

# 1) CRIM: per capita crime rate by town
# 2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3) INDUS: proportion of non-retail business acres per town
# 4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
# 6) RM: average number of rooms per dwelling
# 7) AGE: proportion of owner-occupied units built prior to 1940
# 8) DIS: weighted distances to five Boston employment centres
# 9) RAD: index of accessibility to radial highways
# 10) TAX: full-value property-tax rate per $10,000 [$/10k]
# 11) PTRATIO: pupil-teacher ratio by town
# 12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13) LSTAT: % lower status of the population

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error

boston = pd.read_csv("boston.csv")

df = boston.copy()

fig, axes = plt.subplots(2, 7, figsize=(18, 6))

for i, column in enumerate(df.columns):  # Get column names and indices
    sns.boxplot(x=df[column], ax=axes[i // 7, i % 7])  # Create a boxplot for each column, iterate through subpots using indices
    axes[i // 7, i % 7].set_title(column)  # Set column name as subplot title

plt.tight_layout()  # Adjust subplots
plt.show()

# Handle outliers in CRIM and B columns

# Remove outliers in CRIM

Q1_CRIM = df["CRIM"].quantile(0.25)
Q3_CRIM = df["CRIM"].quantile(0.75)
IQR_CRIM = Q3_CRIM - Q1_CRIM

lower_bound_CRIM = Q1_CRIM - 1.5 * IQR_CRIM
upper_bound_CRIM = Q3_CRIM + 1.5 * IQR_CRIM

outliers_CRIM = (df["CRIM"] < lower_bound_CRIM) | (df["CRIM"] > upper_bound_CRIM)

# Replace outliers in B with mean

Q1_B = df["B"].quantile(0.25)
Q3_B = df["B"].quantile(0.75)
IQR_B = Q3_B - Q1_B

lower_bound_B = Q1_B - 1.5 * IQR_B
upper_bound_B = Q3_B + 1.5 * IQR_B

outliers_B = (df["B"] < lower_bound_B) | (df["B"] > upper_bound_B)

df.loc[outliers_B, "B"] = df["B"].mean()

cleaned_df = df[~outliers_CRIM]

# aykiri_df = df[outliers_B]
# print(f"Outliers: \n{aykiri_df["B"].head(10)}")
# print(f"Outlier analysis: \n{aykiri_df["B"].describe().T}")

# print(f"Dataset: \n{boston.head(5)}")
print(f"Dataset information: \n{boston.describe().T}")

print(f"After cleaning: \n{cleaned_df.describe().T}")
# print(f"Data after cleaning: \n{cleaned_df.head(10)}")


# CORRELATION OF CRIM WITH OTHER VARIABLES

# corr_matrix = cleaned_df.corr() # Calculate correlation matrix

# corr_with_crim = corr_matrix["CRIM"].sort_values(ascending=False) # Correlation values between CRIM and other variables

# print("Correlations with CRIM:")
# print(corr_with_crim)



# SIMPLE LINEAR REGRESSION (TAX - CRIM RELATIONSHIP)

def simple_linear_regression(cleaned_df, independent_var, dependent_var):

    X = cleaned_df[independent_var]
    y = cleaned_df[dependent_var]

    X = sm.add_constant(X) # Constant term

    model = sm.OLS(y, X).fit() # Model fitting

    print("Simple Linear Regression Results ({0} - {1}): ".format(independent_var, dependent_var))
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_fit = sm.OLS(y_train, X_train).fit()

    y_pred = model_fit.predict(X_test)

    r2_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nSimple Linear Regression")
    print("R-Squared: ", r2_squared)
    print("RMSE: ", rmse)


# MULTIPLE LINEAR REGRESSION (RAD AND TAX - CRIM RELATIONSHIP)

def multiple_linear_regression(cleaned_df, independent_vars, dependent_var):

    X = cleaned_df[independent_vars]
    y = cleaned_df[dependent_var]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("Multiple Linear Regression Results: ({0}): ".format(", ".join(independent_vars)))
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_fit = sm.OLS(y_train, X_train).fit()

    y_pred = model_fit.predict(X_test)

    plt.figure(figsize=(12, 6))

    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted', marker='x')
    plt.xlabel('Index')
    plt.ylabel('CRIM')
    plt.title('Actual vs Predicted CRIM (Index-wise)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    r2_squared = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nMultiple Linear Regression")
    print("R-Squared: ", r2_squared)
    print("RMSE: ", rmse)
    


simple_linear_regression(cleaned_df, "TAX", "CRIM")
multiple_linear_regression(cleaned_df, ["RAD", "TAX"], "CRIM")