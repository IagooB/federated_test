import csv
from functools import lru_cache
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import flwr
from confg.configuracion import *

# Paths and caches
_EXPERIMENT_CSV = Path(exper_config.get("experiment_log_path", "datos_experimento.csv"))
_logged_clients = set()


def _initialize_experiment_log():
    _EXPERIMENT_CSV.parent.mkdir(exist_ok=True)
    if not _EXPERIMENT_CSV.exists():
        with open(_EXPERIMENT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["df_shape", "client_id", "slice", "num_samples", "label_counts"])
    else:
        df_log = pd.read_csv(_EXPERIMENT_CSV)
        _logged_clients.update(df_log["client_id"].astype(int).tolist())


_initialize_experiment_log()


def _log_experiment(df_shape, client_id, slice_label, num_samples, label_counts_str):
    if client_id in _logged_clients:
        return
    with open(_EXPERIMENT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([df_shape, client_id, slice_label, num_samples, label_counts_str])
    _logged_clients.add(client_id)


@lru_cache(maxsize=1)
def load_full_dataset(path: str, fraction: float) -> pd.DataFrame:
    # Cargar el dataset y eliminar filas con valores nulos
    df = pd.read_csv(path).dropna().sample(frac=fraction, random_state=SEED)

    if exper_config["excluir_categoria"] is not None:
        df = df[df["Label"] != exper_config["excluir_categoria"]]

    # Codificar la columna 'Label' usando LabelEncoder
    le = preprocessing.LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    # Asegurarse de que la columna 'Slice' sea de tipo entero
    df["Slice"] = df["Slice"].astype(int)

    # Obtener la lista de columnas de características
    all_cols = exper_config["feature_columns"]

    # Si no se usan todas las columnas, eliminar las especificadas
    if not exper_config["todas_columnas"]:
        columns_to_remove = {'Src IP', 'Dst IP', 'Src Port', 'Dst Port'}
        all_cols = [col for col in all_cols if col not in columns_to_remove]

    # Seleccionar solo las columnas relevantes del DataFrame
    df = df[all_cols + ["Slice", "Label"]]

    return df.reset_index(drop=True)


def partition_data(df: pd.DataFrame, num_partitions: int) -> list[pd.DataFrame]:
    """
    Divide el DataFrame en 'num_partitions' partes, asignando
    las primeras num_partitions/2 a Slice==0 y las restantes a Slice==1,
    usando solo df.iloc para partir.
    """
    # 1) Obtener valores únicos de Slice (e.g. [0,1])
    slices = sorted(df["Slice"].unique())
    n_slices = len(slices)
    clients_per_slice = num_partitions // n_slices

    # 2) Preparar lista para cada cliente
    client_dfs: list[pd.DataFrame] = [pd.DataFrame()] * num_partitions

    # 3) Para cada slice, barajar y repartir por iloc
    for idx, slice_val in enumerate(slices):
        # Filtrar y mezclar
        df_slice = (
            df[df["Slice"] == slice_val]
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )
        total = len(df_slice)
        # Tamaño base y resto de la división
        base_size, remainder = divmod(total, clients_per_slice)

        start = 0
        for j in range(clients_per_slice):
            # Distribuir el resto (+1) a las primeras 'remainder' particiones
            size = base_size + (1 if j < remainder else 0)
            end = start + size
            client_id = idx * clients_per_slice + j
            # Partición por iloc
            client_dfs[client_id] = df_slice.iloc[start:end].reset_index(drop=True)
            start = end

    return client_dfs


def load_data(partition_id: int, num_partitions: int):
    # 1) Cargar dataset completo y preprocesar
    full_df = load_full_dataset(
        path=exper_config["data_file_path"],
        fraction=exper_config["fraction"],
    )

    # 2) Particionar: clientes 0–4 Slice=0, 5–9 Slice=1
    client_partitions = partition_data(full_df, num_partitions)
    client_df = client_partitions[partition_id]

    # 3) Determinar slice_label (ahora único por cliente)
    slice_label = int(client_df["Slice"].iloc[0])

    # 4) Log experiment
    df_shape = f"{full_df.shape[0]},{full_df.shape[1]}"
    num_samples = len(client_df)
    vc = client_df["Label"].value_counts().sort_index()
    label_counts_str = ";".join(f"{lab}:{cnt}" for lab, cnt in vc.items())
    _log_experiment(df_shape, partition_id, slice_label, num_samples, label_counts_str)

    # 5) Crear DataLoaders
    #X = client_df.drop(columns=["Label", "Slice"]).values
    X = client_df.drop(columns=["Label"]).values
    y = client_df["Label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=exper_config["test_size"], random_state=SEED
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.long))
    ds_te = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                          torch.tensor(y_te, dtype=torch.long))
    loader_tr = DataLoader(ds_tr,
                           batch_size=exper_config["batch_size"],
                           shuffle=True,
                           drop_last=True)
    loader_te = DataLoader(ds_te,
                           batch_size=exper_config["batch_size"],
                           shuffle=False)

    return loader_tr, loader_te, slice_label


# ---------------- Entrenamiento y evaluación ----------------

def train(model, train_loader, num_epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=exper_config["lr_cliente_simulado"])
    model.train()
    for _ in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            _, preds = outputs.max(1)
            total_correct += (preds == y_batch).sum().item()
            total_count += y_batch.size(0)
    return total_loss / total_count, total_correct / total_count


def get_weights(model: torch.nn.Module) -> flwr.common.Parameters:
    # Extrae TODO el state_dict (parámetros + buffers)
    state_dict = model.state_dict()
    arrays = [tensor.cpu().numpy() for tensor in state_dict.values()]
    return ndarrays_to_parameters(arrays)


def set_weights(model: torch.nn.Module, params: flwr.common.Parameters) -> None:
    # Reconstruye el state_dict completo a partir de los arrays recibidos
    arrays = parameters_to_ndarrays(params)
    keys = list(model.state_dict().keys())
    new_state = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(keys, arrays)
    )
    model.load_state_dict(new_state, strict=True)
