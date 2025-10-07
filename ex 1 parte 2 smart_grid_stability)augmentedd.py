import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ===========================================================
# 1. Carregar o conjunto de dados
# ===========================================================
df = pd.read_csv("smart_grid_stability_augmented.csv")
print("✅ Dados carregados com sucesso!")
print(df.head(), "\n")

# ===========================================================
# 2. Separar variáveis independentes (X) e dependente (y)
# ===========================================================
# A coluna 'stabf' é o alvo (Stable/Unstable)
X = df.drop(columns=["stabf"])
y = df["stabf"]

# Converter o target em 0 (estável) e 1 (instável)
y = y.map({"stable": 0, "unstable": 1})

# ===========================================================
# 3. Dividir em treino e teste
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================================================
# 4. Padronizar as variáveis numéricas
# ===========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================================================
# 5. Criar e treinar modelos
# ===========================================================
modelos = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42)
}

# Dicionário para armazenar resultados
resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    resultados[nome] = {"Acurácia": acc, "F1-score": f1, "Matriz": cm}

    print("=" * 60)
    print(f"🔹 Modelo: {nome}")
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=["Estável", "Instável"]))

# ===========================================================
# 6. Comparar resultados
# ===========================================================
print("\n" + "=" * 60)
print("🔸 COMPARAÇÃO FINAL DOS MODELOS 🔸\n")
df_resultados = pd.DataFrame(resultados).T
print(df_resultados[["Acurácia", "F1-score"]])

melhor = df_resultados["F1-score"].idxmax()
print(f"\n✅ Modelo mais confiável para detectar instabilidade: {melhor}")