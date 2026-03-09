import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


def prepare_data():
    print("Cargando dataset original...")
    df = pd.read_csv('data/train_test_network.csv')

    #Seleccionar métricas puramente numéricas para evitar colapsar la RAM de 512MB
    numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
                    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']

    print("Procesando características...")
    X = df[numeric_cols].fillna(0)
    y = df['label'].astype(int)

    #Escalar los datos para que la regresión logística funcione correctamente
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)
    df_clean = pd.concat([X_scaled, y], axis=1)

    #Guardar dataset centralizado para la evaluación final
    df_clean.to_csv('data/dataset_clean.csv', index=False)

    print("Creando particiones Non-IID (desbalanceadas)...")
    df_normal = df_clean[df_clean['label'] == 0]
    df_attack = df_clean[df_clean['label'] == 1]

    #Definir el grado de asimetría para cada uno de los 5 nodos
    proportions = [(0.95, 0.05), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75), (0.05, 0.95)]
    samples_per_node = 20000  #Límite manejable para no demorar la emulación

    for i, (p_norm, p_att) in enumerate(proportions):
        n_norm = int(samples_per_node * p_norm)
        n_att = int(samples_per_node * p_att)

        part_norm = df_normal.sample(n=n_norm, replace=True)
        part_att = df_attack.sample(n=n_att, replace=True)

        part_final = pd.concat([part_norm, part_att]).sample(frac=1).reset_index(drop=True)
        part_final.to_csv(f'data/client_{i + 1}.csv', index=False)
        print(f"Nodo {i + 1} guardado: {n_norm} benignos, {n_att} ataques.")


if __name__ == "__main__":
    prepare_data()