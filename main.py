import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


df = pd.read_csv("online_shoppers_intention.csv")

df_encoded = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes(bool).columns})

#Zamiana wartości bool (True/False) na liczby (1/0)

print(df_encoded.shape)
print(df_encoded.head())

def walidacja():
    X = df_encoded.drop('Revenue', axis=1)

    y = df_encoded['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=11)

    return X_train, X_test, y_train,y_test

X_train, X_test, y_train,y_test = walidacja()

print(f"Rozmiar zbioru treningowego: {X_train.shape}")
print(f"Rozmiar zbioru testowego: {X_test.shape}")


# Test walidacji
size_train = len(X_train)/(len(X_train)+len(X_test))

assert round(size_train,2) == 0.50 , f"Probka treningowa: {round(size_train,2)}"


