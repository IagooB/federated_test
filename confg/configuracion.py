import os
from pathlib import Path

SEED = 42


def load_config():
    """Load and return experiment configuration."""
    modo_binario = os.getenv("MODO_BINARIO", "False") == "True"

    cfg = {
        "modo_binario": modo_binario,
        "fraction": 1,
        "test_size": 0.2,
        # En binario no necesitamos excluir ninguna clase
        "excluir_categoria": None if modo_binario else "icmp_flood",
        "todas_columnas": True,
        "feature_columns": [
            "Src IP",
            "Dst IP",
            "Src Port",
            "Dst Port",
            "Flow Duration",
            "Total Fwd Packet",
            "Fwd Packet Length Std",
            "ACK Flag Count",
            "Fwd Seg Size Min",
            "Protocol",
        ],
        "con_Slice": True,
        "clientes_random": True,
        "num_rounds": 100,
        "num_clients": 10,
        "clients_por_ronda": 5,
        "fraction_fit": 1.0,
        "local_epochs": 1,
        "batch_size": 32,
        "lr_cliente_simulado": 0.01,
        # Paths
        "client_info_dir": "data/client_info",
        "experiment_log_path": "data/datos_experimento.csv",
        "metrics_log_path": "metricas/metricas_globales.csv",
        "client_metrics_log_path": "metricas/metricas_por_cliente.csv",
        "wandb_project": "flower-federated",
        "wandb_project_name": "flower-federated",
    }

    # Ajustes según modo de operación
    if modo_binario:
        cfg["data_file_path"] = (
            r"D:\TRABALLO_CODE\data\combinada\binario\federado\bi_fed.csv"
        )
        cfg["num_clases"] = 2
    else:
        cfg["data_file_path"] = (
            r"D:\TRABALLO_CODE\data\combinada\multi\federado\multi_fed.csv"
        )
        cfg["num_clases"] = 7 - (
            1 if cfg["excluir_categoria"] is not None else 0
        )

    # Calculamos la dimensión de entrada
    if cfg["todas_columnas"]:
        cfg["shape_entrada"] = len(cfg["feature_columns"]) + 1
    else:
        cfg["shape_entrada"] = (
            len(cfg["feature_columns"])  # type: ignore
            - len(["Src IP", "Dst IP", "Src Port", "Dst Port"])  # type: ignore
            + 1
        )

    # Si no usamos Slice, reducimos input en 1
    if not cfg["con_Slice"]:
        cfg["shape_entrada"] -= 1

    # Nombre dinámico de guardado de pesos
    cols = "all" if cfg["todas_columnas"] else "noIPPorts"
    slc = "withSlice" if cfg["con_Slice"] else "noSlice"
    rand = "rand" if cfg["clientes_random"] else "fixed"
    lr_str = str(cfg["lr_cliente_simulado"]).replace(".", "p")
    filename = f"pesos/global_{cols}_{slc}_{rand}_lr{lr_str}.csv"
    cfg["guardado_pesos"] = Path(filename)

    return cfg


# Configuración cargada y forma de entrada
exper_config = load_config()
input_shape = exper_config["shape_entrada"]
