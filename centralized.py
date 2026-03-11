import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("Entrenando Modelo Centralizado Base...")

df = pd.read_csv('data/dataset_clean.csv')

#Aislar las características (X) y las etiquetas de intrusion (y)
X = df.drop(columns=['label']).values
y = df['label'].values

#Divdir los datos en subconjuntos de entrenamiento y prueba (80/20).
#Fijar la semilla aleatoria para garantizar la reproducibilidad del experimento.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Establecer el límite de convergencia en 100 iteraciones sobre los datos
model = LogisticRegression(max_iter=100, solver="saga") #optimizador 'saga'

#Entrenar modelo
model.fit(X_train, y_train)

#Predicciones con l conjunto de prueba aislado.
y_pred = model.predict(X_test)


print(f"Accuracy Centralizado: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score Centralizado: {f1_score(y_test, y_pred):.4f}")