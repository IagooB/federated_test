SEED = 42


def load_config():
    """
    Carga los parámetros de configuración para el experimento de clasificación multiclase (7 categorías).
    """
    return {
        # Muestreo y partición
        "fraction": 0.1,  # Fracción del dataset completo a muestrear
        "test_size": 0.2,  # Proporción de test/training en cada partición
        "excluir_categoria": None,  # No excluimos ninguna clase en multiclase
        "todas_columnas": True,  # Usar solo feature_columns, no todas las columnas

        # Columnas de features
        "feature_columns": [
            "Src IP", "Dst IP", "Src Port", "Dst Port",
            "Flow Duration", "Total Fwd Packet",
            "Fwd Packet Length Std", "ACK Flag Count",
            "Fwd Seg Size Min", "Protocol"
        ],

        # Federated Learning
        "clientes_random": True,
        "num_rounds": 3,  # Número de rondas de federado
        "num_clients": 10,  # Total de clientes
        "clients_por_ronda": 5,  # Clientes muestreamados por ronda
        "fraction_fit": 1.0,  # Fracción de clientes que harán fit
        "local_epochs": 5,  # Épocas locales por cliente
        "batch_size": 32,  # Tamaño de batch en DataLoader
        "lr_cliente_simulado": 0.01,  # Learning rate del optimizador local

        # Datos públicos compartidos (solo si public_fraction > 0)
        "public_fraction": 0.1,  # Fracción de datos para dataset público
        "public_label": 1,  # Etiqueta que siempre incluimos en público

        # Rutas y logs
        "data_file_path": "D:/TRABALLO_CODE/data/combinada/multi/federado/multi_fed.csv",
        "client_info_dir": "data/client_info",  # Misma carpeta que en binario
        "experiment_log_path": "data/datos_experimento.csv",  # CSV de metadatos de experimento
        "guardado_pesos": "pesos/global_data_multiclase.csv",  # CSV de cambios de pesos
        "metrics_log_path": "metricas/metricas_globales.csv",  # CSV de métricas agregadas
        "client_metrics_log_path": "metricas/metricas_por_cliente.csv",  # CSV de métricas por cliente

        # Weights & Biases
        "wandb_project": "flower-federated",
        "wandb_project_name": "flower-federated",
    }


# Carga la configuración y calcula el input_shape a partir de feature_columns
exper_config = load_config()
input_shape = len(exper_config["feature_columns"])
