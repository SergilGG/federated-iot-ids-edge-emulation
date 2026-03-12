import flwr as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys
import utils
import warnings

warnings.filterwarnings("ignore")

client_id = sys.argv[1]
print(f"Iniciando Nodo Edge {client_id}...")

df = pd.read_csv(f'data/client_{client_id}.csv')
X_train = df.drop(columns=['label']).values
y_train = df['label'].values

model = LogisticRegression(
    penalty="l2",
    max_iter=1,
    warm_start=True, #No olvidar el conocimiento aprendido entre rondas
    solver="saga"
)
utils.set_initial_params(model, n_features=X_train.shape[1], n_classes=2)

class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):
        utils.set_model_params(model, parameters)
        model.fit(X_train, y_train) #Entrena localmente con sus datos
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_train, model.predict_proba(X_train))
        accuracy = model.score(X_train, y_train)
        return float(loss), len(X_train), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_client(server_address="server:8080", client=IDSClient())