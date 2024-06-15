# Predicting Car Prices in USA

# Comparison of Ridge, Lasso and ElasticNet Regression Models

# Dataset: US Cars Dataset

# Goal: Predicting the car prices based on their features

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

vehicles = pd.read_csv("USA_cars_datasets.csv")

df = vehicles.copy()

print(df.describe().T)

# Cleaning

df = df.drop(columns=["Unnamed: 0"]) # Dropping the unnamed column

numeric_df = df.select_dtypes(include=["int64","float64"])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i, column in enumerate(numeric_df.columns):
    sns.boxplot(x=numeric_df[column], ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(column)

plt.tight_layout()
plt.show()

# Mileage outliers

Q1_mileage = numeric_df["mileage"].quantile(0.25)
Q3_mileage = numeric_df["mileage"].quantile(0.75)
IQR_mileage = Q3_mileage - Q1_mileage

lower_bound_mileage = Q1_mileage - 1.5 * IQR_mileage
upper_bound_mileage = Q3_mileage + 1.5 * IQR_mileage

outliers_mileage = (numeric_df["mileage"] < lower_bound_mileage) | (numeric_df["mileage"] > upper_bound_mileage)

# Price outliers

Q1_price = numeric_df["price"].quantile(0.25)
Q3_price = numeric_df["price"].quantile(0.75)
IQR_price = Q3_price - Q1_price

lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

outliers_price = (numeric_df["price"] < lower_bound_price) | (numeric_df["price"] > upper_bound_price)

# Mileage & Price

filtered_rows = df[(outliers_price) | (outliers_mileage) | (df["price"] < 500)]

print("Filtered Rows:")
print(filtered_rows.count())

print("Before Cleaning")
print(numeric_df.describe().T)

df = df.drop(filtered_rows.index)

print("After Cleaning")
print(df.describe().T)

# Cleaning done

def encode_categorical_features(df, columns):
    
    for col in columns:
        freq = df[col].value_counts()
        seq = freq.index.tolist()
        numeric_dict = {val: i + 1 for i, val in enumerate(seq)}
            
        new_col_name = f"{col}_N"
        df[new_col_name] = df[col].map(numeric_dict)

    return df


columns_to_encode = ["brand", "model", "color"]
df_encoded = encode_categorical_features(df, columns_to_encode)

print(df_encoded.head(10))

df = df_encoded

# Correlations 

corr_numeric_df = df.select_dtypes(include=["int64","float64"])
corr_matrix = corr_numeric_df.corr()
corr_with_price = corr_matrix["price"].sort_values(ascending=False)
print("Correlations with price: ")
print(corr_with_price)

# Modeling

y = df["price"]
#X = df.drop(columns=["price","brand","model","title_status","color","vin","lot","state","country","condition"])
X = df.drop(columns=["price","brand","model","title_status","color","vin","state","country","condition"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Ridge Regression Model

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_r2 = r2_score(y_test, y_pred_ridge)

print("\nRidge RMSE: ", ridge_rmse)
print("Ridge R2 score: ", ridge_r2)


# Lasso Regression Model

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
lasso_r2 = r2_score(y_test, y_pred_lasso)

print("\nLasso RMSE: ", lasso_rmse)
print("Lasso R2 score: ", lasso_r2)


# ElasticNet Regression Model

from sklearn.linear_model import ElasticNet, ElasticNetCV

enet_model = ElasticNet()
enet_model.fit(X_train, y_train)

y_pred_enet = enet_model.predict(X_test)
enet_rmse = np.sqrt(mean_squared_error(y_test, y_pred_enet))
enet_r2 = r2_score(y_test, y_pred_enet)

print("\nElasticNet RMSE: ", enet_rmse)
print("ElasticNet R2 score: ", enet_r2)

# This dataset did not perform well with ridge, lasso and elasticNet regression models
