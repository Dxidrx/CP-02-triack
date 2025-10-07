import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Carregar dataset ---
df = pd.read_csv("SolarPrediction.csv")

# Converter colunas de data/hora (se existirem)
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
            df[col + "_hour"] = df[col].dt.hour
            df[col + "_day"] = df[col].dt.day
            df[col + "_month"] = df[col].dt.month
            df.drop(columns=[col], inplace=True)
        except:
            pass

# --- 2. Criar variável-alvo (Alta/Baixa radiação) ---
limiar = df["Radiation"].median()
df["Classe"] = df["Radiation"].apply(lambda x: "Alta" if x > limiar else "Baixa")

# --- 3. Separar atributos e alvo ---
X = df.drop(columns=["Radiation", "Classe"])
y = df["Classe"]

# Garantir que só haja colunas numéricas
X = X.select_dtypes(include=["float64", "int64"])

# --- 4. Separar treino e teste (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 5. Normalizar dados ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Treinar e avaliar modelos ---
modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

print("=== CLASSIFICAÇÃO DE RADIAÇÃO SOLAR ===")
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    print(f"\nModelo: {nome}")
    print("Acurácia:", round(accuracy_score(y_test, y_pred), 3))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

