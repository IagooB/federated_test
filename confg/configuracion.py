import os
from pathlib import Path

SEED = 42


def load_config():
    # Lectura de variable de entorno o valor por defecto
    modo_binario = os.getenv("MODO_BINARIO", "False") == "True"

    # Configuración base (la mayoría de valores iguales en ambos modos)
    cfg = {
        "modo_binario": modo_binario,
        "fraction": 1,
        "test_size": 0.2,
        # En binario no necesitamos excluir ninguna clase, el CSV ya contiene 2 clases
        "excluir_categoria": None if modo_binario else "icmp_flood",
        "todas_columnas": False,
        "shape_entrada": 11,
        "feature_columns": [
            "Src IP", "Dst IP", "Src Port", "Dst Port",
            "Flow Duration", "Total Fwd Packet",
            "Fwd Packet Length Std", "ACK Flag Count",
            "Fwd Seg Size Min", "Protocol"
        ],
        "clientes_random": False,
        "num_rounds": 100,
        "num_clients": 10,
        "clients_por_ronda": 5,
        "fraction_fit": 1.0,
        "local_epochs": 5,
        "batch_size": 32,
        "lr_cliente_simulado": 0.01,
        "public_fraction": 0.1,
        "public_label": 1,
        "client_info_dir": "data/client_info",
        "experiment_log_path": "data/datos_experimento.csv",
        "guardado_pesos": "pesos/global_data.csv",
        "metrics_log_path": "metricas/metricas_globales.csv",
        "client_metrics_log_path": "metricas/metricas_por_cliente.csv",
        "wandb_project": "flower-federated",
        "wandb_project_name": "flower-federated",
    }

    # Ahora ajustamos según el modo
    if modo_binario:
        # Dataset y número de clases para binario
        cfg["data_file_path"] = r"D:\TRABALLO_CODE\data\combinada\binario\federado\bi_fed.csv"
        cfg["num_clases"] = 2
    else:
        # Dataset multiclase
        cfg["data_file_path"] = r"D:\TRABALLO_CODE\data\combinada\multi\federado\multi_fed.csv"

        # Aquí partimos de 7 clases totales (0–6)
        cfg["num_clases"] = 7 - (1 if cfg["excluir_categoria"] is not None else 0)

    if cfg["todas_columnas"]:
        cfg["shape_entrada"] = len(cfg["feature_columns"]) + 1
    else:
        cfg["shape_entrada"] = (len(cfg["feature_columns"])) - (len(["Src IP", "Dst IP", "Src Port", "Dst Port"])) + 1
    return cfg


# Cargamos así la configuración
exper_config = load_config()
input_shape = exper_config["shape_entrada"]
