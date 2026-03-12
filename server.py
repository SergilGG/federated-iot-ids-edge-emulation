import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5, #al menos 5 dispositivos participqn en el entrenamiento
        min_evaluate_clients=5, #al menos 5 dispositivos participan en la evaluación
        min_available_clients=5, #obligar a que no empiece a hacer los promedios (FedAvg) hasta que los 5 nodos edge de docker esten vivos y conectados
    )
    print("Iniciando Servidor Federado...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100), # n rondas de entrenamiento
        strategy=strategy,
    )