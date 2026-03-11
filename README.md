# Detección Distribuida de Intrusiones IoT mediante Federated Learning

Este repositorio contiene la implementación de un Sistema de Detección de Intrusiones (IDS) descentralizado utilizando Aprendizaje Federado (Federated Learning). 

El proyecto emula un entorno *Edge Computing* restringiendo los recursos de los nodos mediante Docker para simular dispositivos IoT (tipo Raspberry Pi), entrenando de forma colaborativa un modelo de Regresión Logística sobre el dataset TON_IoT particionado de manera asimétrica (Non-IID).

## Tecnologías
* **Python 3.10**
* **Flower (`flwr`)**: Framework para el Aprendizaje Federado.
* **Scikit-Learn**: Entrenamiento del modelo de Regresión Logística (con inicialización `warm_start`).
* **Docker & Docker Compose**: Manejo y emulación de hardware (`cgroups`).
* **Pandas & NumPy**: Procesamiento del dataset.

## Estructura del Proyecto

```text
.
├── data/                   # Carpeta para los datasets (original, limpio y particiones)
├── centralized.py          # Modelo base para comparación
├── client.py               # Lógica del nodo edge (entrenamiento local)
├── server.py               # Servidor (estrategia FedAvg)
├── preprocess.py           # Limpieza y partición Non-IID del dataset
├── utils.py                # Adaptadores de extracción de pesos de Scikit-Learn para Flower
├── docker-compose.yml      # Manejo de la red y limitador de recursos (1 CPU, 512MB RAM)
├── Dockerfile              # Imagen base de los contenedores
└── requirements.txt        # Dependencias de Python
```

## Ejecución

1. Limpia el dataset original, con valores únicamente numéricos, y ese arhicvo se particiona en  en 5 subarchivos Non-IID.
``` code
pip install -r requirements.txt
python preprocess.py
```

2. Iniciar infraestructura (1 Servidor + 5 Nodos Edge).
``` code
docker-compose up --build
```

3. Ejecución del modelo centralizado (para la comparación)
``` code
python centralized.py
```

## Resultados esperados

Al ejecutar el entorno federado con Docker, el servidor arrojará un resumen donde la pérdida logarítmica (Loss) distribuida debe reducirse constantemente a través de las 5 rondas de comunicación (iniciando alrededor de 0.6869 y descendiendo hasta 0.6711 aprox.), confirmando que la federación de nodos aprende exitosamente sin compartir sus datos locales.
