import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# 1. Ingesta de datos (simulada)
np.random.seed(42)
data = pd.DataFrame({
    'temperatura': np.random.normal(4, 2, 100),
    'humedad': np.random.normal(80, 10, 100),
})

# Regla sencilla: fallo si temperatura > 5 y humedad < 75
data['fallo'] = ((data['temperatura'] > 5) & (data['humedad'] < 75)).astype(int)

# 2. Procesamiento - Filtro
data.dropna(inplace=True)
data = data.astype({'temperatura': 'float', 'humedad': 'float', 'fallo': 'int'})

# 3. EDA (opcional)
sns.set(style="ticks", font_scale=1.2)
g = sns.pairplot(data, hue='fallo', height=4)
for i, varname in enumerate(data.columns[:-1]):
    g.axes[0, i].set_title(varname, fontsize=16, pad=20)
for i in range(len(data.columns) - 1):
    for j in range(len(data.columns) - 1):
        ax = g.axes[i, j]
        ax.set_xlabel(data.columns[j], fontsize=12)
        ax.set_ylabel(data.columns[i], fontsize=12)
        ax.tick_params(axis='x', labelbottom=True)
        ax.tick_params(axis='y', labelleft=True)
plt.subplots_adjust(top=0.9, wspace=0.4, hspace=0.4)
plt.show()

# 4. Ingeniería de características
data['temp_var'] = data['temperatura'].rolling(window=3).std().fillna(0)

# 5. Entrenamiento
X = data[['temperatura', 'humedad', 'temp_var']]
y = data['fallo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo_path = "modelo/xgboost_model.pkl"
fallos_reales_path = "modelo/fallos_reales.csv"

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    print("Modelo XGBoost cargado desde disco.")

    # Guardar fallos reales del test
    fallos_reales = X_test[y_test == 1]
    if not fallos_reales.empty:
        if os.path.exists(fallos_reales_path):
            df_fallos_reales = pd.read_csv(fallos_reales_path)
            # Quitar columnas vacías o solo NaN antes de concatenar
            df_fallos_reales = df_fallos_reales.dropna(axis=1, how='all')
        else:
            df_fallos_reales = pd.DataFrame(columns=fallos_reales.columns)

        # También limpiar fallos_reales por si acaso
        fallos_reales = fallos_reales.dropna(axis=1, how='all')

        df_fallos_reales = pd.concat([df_fallos_reales, fallos_reales], ignore_index=True).drop_duplicates().reset_index(drop=True)
        if len(df_fallos_reales) > 100:
            df_fallos_reales = df_fallos_reales.tail(100)
        df_fallos_reales.to_csv(fallos_reales_path, index=False)
        print(f"Fallos reales guardados: {len(df_fallos_reales)}")


        # Reentrenar modelo con fallos reales
        y_fallos = pd.Series([1]*len(df_fallos_reales))
        X_train_actualizado = pd.concat([X_train, df_fallos_reales])
        y_train_actualizado = pd.concat([y_train, y_fallos])
        model.fit(X_train_actualizado, y_train_actualizado)
        joblib.dump(model, modelo_path)
        print("Modelo reentrenado con fallos reales y guardado.")
    else:
        print("No hay fallos reales nuevos para guardar ni reentrenar.")

else:
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    os.makedirs("modelo", exist_ok=True)
    joblib.dump(model, modelo_path)
    print("Modelo XGBoost entrenado y guardado.")

# 6. Evaluación
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

metricas_path = "modelo/historico_metricas.csv"
registro = pd.DataFrame([{
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "total_test": len(y_test),
    "fallos_reales": sum(y_test),
    "fallos_predichos": sum(y_pred)
}])

if os.path.exists(metricas_path):
    registro.to_csv(metricas_path, mode='a', header=False, index=False)
else:
    registro.to_csv(metricas_path, mode='w', header=True, index=False)

# 7. Predicción ejemplo
new_data = pd.DataFrame({
    'temperatura': [6],
    'humedad': [74],
    'temp_var': [0.3]
})

pred = model.predict(new_data)
print("¿Fallará?:", bool(pred[0]))