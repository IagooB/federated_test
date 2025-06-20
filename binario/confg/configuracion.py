#### BINARIO ####

SEED = 42


def load_config():
    return {
        # Modelo
        "fraction": 1,
        "test_size": 0.2,
        "excluir_categoria": None,
        "todas_columnas": True,

        # Estructura de datos
        "feature_columns": [
            'Src IP', 'Dst IP', 'Src Port', 'Dst Port',
            'Flow Duration', 'Fwd Packet Length Std',
            'ACK Flag Count', 'Protocol', 'Total Fwd Packet',
            'Fwd Seg Size Min'
        ],
        "slice_column": "Slice",
        "label_column": "Label",

        # Federado
        "num_rounds": 200,
        "num_clients": 10,
        "clientes_random": True,
        "clients_por_ronda": 5,
        "fraction_fit": 1.0,
        "local_epochs": 3,
        "batch_size": 32,
        "lr_cliente_simulado": 0.01,

        # Paths
        "data_file_path": "D:/TRABALLO_CODE/data/combinada/binario/federado/bi_fed.csv",
        "client_info_dir": "data/client_info",
        "weights_dir": "pesos",
        "guardado_pesos": "pesos/pesos_fl_bi.csv",

        # Logging
        "experiment_log_path": "data/datos_experimento.csv",
        "metrics_log_path": "metricas/metricas_globales.csv",
        "client_metrics_log_path": "metricas/metricas_por_cliente.csv"
    }


exper_config = load_config()
input_shape = len(exper_config["feature_columns"])
