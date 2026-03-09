import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("Entrenando Modelo Centralizado Base...")
df = pd.read_csv('data/dataset_clean.csv')
X = df.drop(columns=['label']).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=100, solver="saga")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy Centralizado: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score Centralizado: {f1_score(y_test, y_pred):.4f}")