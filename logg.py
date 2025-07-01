import os
import logging
from confg.configuracion import exper_config
import pandas as pd
import numpy as np

#### LOG ####
LOG_DIR = '../logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'execution.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def log_weight_update(weights_before, weights_after, label, csv_filename):
    """
    Registra en un CSV la actualización de pesos.

    Args:
        weights_before (list of np.array): Pesos antes de la actualización.
        weights_after (list of np.array): Pesos después de la actualización.
        label (str): Etiqueta de Slice (por ejemplo, "Slice1" o "Slice0").
        csv_filename (str): Ruta del archivo CSV donde se almacenará el registro.
    """
    print(f"Logueando actualización para etiqueta {label} en {csv_filename}")

    # Función interna para aplanar la lista de arrays en un único vector
    def flatten_weights(weights):
        return np.concatenate([w.flatten() for w in weights])

    w_before = flatten_weights(weights_before)
    w_after = flatten_weights(weights_after)
    diff = w_after - w_before

    # Preparar el registro como un DataFrame
    data = {
        "pesos_antes": [w_before.tolist()],
        "pesos_despues": [w_after.tolist()],
        "diferencia": [diff.tolist()],
        "etiqueta": [label]
    }
    df_new = pd.DataFrame(data)

    # Si el archivo ya existe se añade, de lo contrario se crea
    if os.path.exists(csv_filename):
        df_new.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_filename, mode='w', header=True, index=False)
