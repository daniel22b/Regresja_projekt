import pandas as pd

df = pd.read_csv("online_shoppers_intention.csv")

def encoded(df):
    """
    Przekształca zmienne kategoryczne na zmienne zero-jedynkowe (one-hot encoding).

    Args:
        df (pd.DataFrame): Surowy DataFrame.

    Returns:
        pd.DataFrame: Zakodowany DataFrame z usunięciem jednej kolumny z każdego zestawu dummy 
                      i konwersją wartości logicznych na liczby całkowite.
    """
    df_encoded = pd.get_dummies(df,drop_first=True)

    df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes(bool).columns})

    return df_encoded


def validacion(data):
    """
    Dzieli dane na zestaw treningowy i testowy.

    Args:
        data (pd.DataFrame): Zakodowany DataFrame zawierający zmienną docelową 'Revenue'.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = data.drop('Revenue', axis=1)

    y = data['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=12)

    return X_train, X_test, y_train,y_test


def vif(X_train):
    """
    Oblicza współczynniki inflacji wariancji (VIF) dla zestawu treningowego.

    Args:
        X_train (pd.DataFrame): Dane treningowe (cechy).

    Returns:
        pd.DataFrame: Tabela z nazwami cech i odpowiadającymi im wartościami VIF.
    """

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X_vif = add_constant(X_train)

    vif_data = pd.DataFrame()

    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    return vif_data

def regularization(X_train, X_test, y_train, y_test, vif_data):
    """
    Usuwa cechy o wysokim VIF, normalizuje dane i trenuje model regresji logistycznej z regularizacją L2.

    Args:
        X_train (pd.DataFrame): Dane treningowe (cechy).
        X_test (pd.DataFrame): Dane testowe (cechy).
        y_train (pd.Series): Etykiety treningowe.
        y_test (pd.Series): Etykiety testowe.
        vif_data (pd.DataFrame): Dane VIF do wyboru cech.

    Returns:
        Tuple: y_pred (predykcje binarne), y_proba (prawdopodobieństwa), y_test (rzeczywiste etykiety testowe).
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    wrong_data = []
    for key,value in vif_data.values:
        if key != 'const' and value > 2:
            wrong_data.append(key)

    X_train_reduced = X_train.drop(columns=wrong_data)
    X_test_reduced = X_test.drop(columns=wrong_data)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)

    model = LogisticRegression(penalty='l2', solver='saga', max_iter=10000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]


    return y_pred, y_proba, y_test


def threshold(y_proba, y_test, threshold = 0.64):
    """
    Oblicza metryki klasyfikacji dla danego progu decyzyjnego.

    Args:
        y_proba (np.ndarray): Prawdopodobieństwa klasy pozytywnej.
        y_test (pd.Series): Rzeczywiste etykiety testowe.
        threshold (float): Próg klasyfikacji.

    Prints:
        Classification report oraz confusion matrix.
    """

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    y_pred_thresh = (y_proba >= threshold).astype(int)

    print(f"Classification Report (threshold={threshold}):")
    print(classification_report(y_test, y_pred_thresh))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_thresh))

if __name__ == '__main__':


    data = encoded(df)
    X_train, X_test, y_train, y_test = validacion(data)
    vif_data = vif(X_train)
    y_pred, y_proba, y_test_real = regularization(X_train, X_test, y_train, y_test, vif_data)
    threshold(y_proba, y_test_real)

    print(f"Rozmiar zbioru treningowego: {X_train.shape}")
    print(f"Rozmiar zbioru testowego: {X_test.shape}")



    # Test walidacji
    size_train = len(X_train)/(len(X_train)+len(X_test))

    assert round(size_train,2) == 0.70 , f"Probka treningowa: {round(size_train,2)}"

