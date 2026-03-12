import flwr as fl


def metrics_average(metrics):
    """
    Promediar el Accuracy y F1-Score recibidos de los nodos Edge.
    """
    if not metrics:
        return {}

    #Sumar el total
    total_ejemplos = sum([num_ejemplos for num_ejemplos, _ in metrics])

    #Calcular los promedios ponderados
    accuracy_promedio = sum([num_ejemplos * m["accuracy"] for num_ejemplos, m in metrics]) / total_ejemplos
    f1_promedio = sum([num_ejemplos * m["f1_score"] for num_ejemplos, m in metrics]) / total_ejemplos

    #Retornar el diccionario con las metricas globales
    return {"accuracy": accuracy_promedio, "f1_score": f1_promedio}


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5,  #al menos 5 dispositivos participqn en el entrenamiento
        min_evaluate_clients=5,  #al menos 5 dispositivos participan en la evaluación
        min_available_clients=5,
        #esperar hasta que los 5 nodos edge de docker esten vivos y conectados
        evaluate_metrics_aggregation_fn=metrics_average  #Inyectar la funcion de promediado
    )
    print("Iniciando Servidor Federado...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=100),  #n rondas de entrenamiento
        strategy=strategy,
    )