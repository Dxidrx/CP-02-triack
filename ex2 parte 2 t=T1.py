
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carregar o dataset
df = pd.read_csv("T1.csv")

# 2. Verificar as colunas disponíveis
print("Colunas do dataset:", list(df.columns))


target = "LV ActivePower (kW)"  

# 4. Separar variáveis explicativas (X) e alvo (y)
X = df.drop(columns=[target])
y = df[target]

# 5. Remover colunas não numéricas
X = X.select_dtypes(include=["float64", "int64"])

# 6. Substituir valores ausentes por média
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 7. Dividir em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Criar e treinar os modelos
modelos = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Regressão": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# 10. Avaliar os modelos
print("\n=== RESULTADOS REGRESSÃO EÓLICA ===")
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nModelo: {nome}")
    print("RMSE:", round(rmse, 2))
    print("R²:", round(r2, 3))
