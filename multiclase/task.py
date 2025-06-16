import csv
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import wandb
from sklearn.metrics import classification_report

from confg.configuracion import *

# Paths and experiment logging cache
_EXPERIMENT_CSV = Path(exper_config.get("experiment_log_path", "datos_experimento.csv"))
_logged_clients = set()
import os


def _initialize_experiment_log():
    _EXPERIMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not _EXPERIMENT_CSV.exists():
        with open(_EXPERIMENT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["df_shape", "client_id", "slice", "num_samples", "label_counts"])
    else:
        df_log = pd.read_csv(_EXPERIMENT_CSV)
        _logged_clients.update(df_log["client_id"].astype(int).tolist())


_initialize_experiment_log()


def _log_experiment(df_shape: str, client_id: int, slice_label: int, num_samples: int, label_counts_str: str):
    if client_id in _logged_clients:
        return
    with open(_EXPERIMENT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([df_shape, client_id, slice_label, num_samples, label_counts_str])
    _logged_clients.add(client_id)


@lru_cache(maxsize=1)
def load_full_dataset(path: str, fraction: float, exclude_label: int = None) -> pd.DataFrame:
    """
    Carga y preprocesa el dataset completo:
    - Muestreo según fraction
    - Opcionalmente excluye una categoría
    - Codifica labels con LabelEncoder
    - Convierte Slice a int
    - Selecciona columnas de features según config
    """
    df = pd.read_csv(path).dropna().sample(frac=fraction, random_state=SEED)
    if exclude_label is not None:
        df = df[df["Label"] != exclude_label]

    le = preprocessing.LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    df["Slice"] = df["Slice"].astype(int)

    if exper_config.get("todas_columnas", False):
        feature_cols = exper_config["feature_columns"]
    else:
        feature_cols = exper_config["feature_columns"]

    # Nos aseguramos de incluir Label y Slice para particionar y loguear
    return df[feature_cols + ["Label", "Slice"]].reset_index(drop=True)


def partition_data(df: pd.DataFrame, num_partitions: int) -> list[pd.DataFrame]:
    """
    Divide df en num_partitions clientes, manteniendo proporción por Slice.
    """
    slices = sorted(df["Slice"].unique())
    n_slices = len(slices)
    clients_per_slice = num_partitions // n_slices
    client_dfs = [pd.DataFrame() for _ in range(num_partitions)]

    for idx, slice_val in enumerate(slices):
        df_slice = df[df["Slice"] == slice_val].sample(frac=1, random_state=SEED).reset_index(drop=True)
        total = len(df_slice)
        base, rem = divmod(total, clients_per_slice)
        start = 0
        for j in range(clients_per_slice):
            size = base + (1 if j < rem else 0)
            end = start + size
            client_id = idx * clients_per_slice + j
            client_dfs[client_id] = df_slice.iloc[start:end].reset_index(drop=True)
            start = end

    return client_dfs


# Cachés para particiones y datos públicos
_partitions_cache: dict[int, pd.DataFrame] = {}
_public_df: pd.DataFrame | None = None


def get_public_data(
        df: pd.DataFrame,
        public_fraction: float,
        label_col: str = "Label",
        random_state: int = 42
) -> pd.DataFrame:
    """
    Toma un DataFrame completo y devuelve un subset público
    sacado de cada clase en proporción `public_fraction`.
    """
    # Lista de DataFrames muestreado por etiqueta
    stratified_samples = []
    for lbl, group in df.groupby(label_col):
        # Para cada clase, muestreamos `public_fraction` de sus filas
        stratified_samples.append(
            group.sample(
                frac=public_fraction,
                replace=False,
                random_state=random_state
            )
        )
    # Concatenar y reordenar
    public_df = pd.concat(stratified_samples, ignore_index=True)
    return public_df.sample(frac=1.0, random_state=random_state)


def compute_class_weights(
        labels: np.ndarray,
        normalize: bool = True
) -> torch.Tensor:
    """
    Dados un array de etiquetas (enteros 0..K-1),
    devuelve un tensor de pesos para CrossEntropyLoss:
    peso[i] ∝ 1/count_i.
    """
    num_classes = 7
    counts = np.bincount(labels, minlength=num_classes)
    # Evitar ceros
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    if normalize:
        # Normalizar para que sumen len(weights)
        weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def load_data(partition_id: int, num_partitions: int):
    """
    Para un cliente dado:
    1. Carga y preprocesa el dataset completo
    2. Crea (una vez) el dataset público si corresponde
    3. Parte los datos entre los clientes
    4. Guarda en disco el CSV local de este cliente
    5. Añade los datos públicos
    6. Loguea meta-datos del experimento
    7. Construye y devuelve DataLoaders (train, test) y slice_label
    8. Inicia un run de W&B para este cliente
    """
    global _partitions_cache, _public_df

    # 1) Dataset completo
    full_df = load_full_dataset(
        exper_config["data_file_path"],
        exper_config["fraction"],
        exper_config.get("excluir_categoria")
    )

    # 2) Dataset público (singleton)
    if exper_config.get("public_fraction", 0) > 0 and _public_df is None:
        _public_df = get_public_data(full_df, exper_config["public_fraction"])
    public_df = _public_df

    # 3) Particiones por cliente
    if not _partitions_cache:
        parts = partition_data(full_df, num_partitions)
        _partitions_cache = {i: parts[i] for i in range(len(parts))}

    client_df = _partitions_cache[partition_id]

    # 4) Guardar CSV del cliente
    #Path("data").mkdir(parents=True, exist_ok=True)
    #client_df.to_csv(f"data/client_{partition_id}_data.csv", index=False)

    # 5) Añadir datos públicos
    if public_df is not None:
        client_df = pd.concat([client_df, public_df], ignore_index=True)
        client_df = client_df.sample(frac=1).reset_index(drop=True)

    # 6) Log experimento
    df_shape = f"{full_df.shape[0]},{full_df.shape[1]}"
    slice_label = int(client_df["Slice"].mode()[0])
    num_samples = len(client_df)
    vc = client_df["Label"].value_counts().sort_index()
    label_counts_str = ";".join(f"{lab}:{cnt}" for lab, cnt in vc.items())
    _log_experiment(df_shape, partition_id, slice_label, num_samples, label_counts_str)

    # 7) Crear DataLoaders
    X = client_df.drop(columns=["Label", "Slice"]).values
    y = client_df["Label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=exper_config["test_size"], random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.long))
    ds_te = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                          torch.tensor(y_te, dtype=torch.long))

    loader_tr = DataLoader(ds_tr, batch_size=exper_config["batch_size"], shuffle=True, drop_last=True)
    loader_te = DataLoader(ds_te, batch_size=exper_config["batch_size"], shuffle=False)

    return loader_tr, loader_te, slice_label


def train(model: torch.nn.Module, train_loader: DataLoader, num_epochs: int = 1):
    """
    Entrena el modelo por num_epochs y loguea métricas de loss y accuracy en W&B.
    """
    # 2) Extrae todas las etiquetas de entrenamiento
    all_labels = []
    for _, y in train_loader:
        all_labels.append(y.numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    class_weights = compute_class_weights(all_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=exper_config["lr_cliente_simulado"])
    model.train()

    for epoch in range(num_epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total


def evaluate(model: torch.nn.Module, test_loader: DataLoader):
    """
    Evalúa el modelo, calcula loss, accuracy y reporte de clasificación,
    todo logueado en W&B.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_count += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    return avg_loss, accuracy


def get_weights(net: torch.nn.Module) -> list:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: torch.nn.Module, parameters: list):
    state_dict = net.state_dict()
    for key, array in zip(state_dict.keys(), parameters):
        state_dict[key] = torch.tensor(array)
    net.load_state_dict(state_dict, strict=True)
