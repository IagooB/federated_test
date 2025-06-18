import json
import logging
import random
from pathlib import Path
from typing import Tuple, List

import flwr as fl
import numpy as np
import pandas as pd
import wandb
from flwr.common import (
    parameters_to_ndarrays,
    GetPropertiesIns,
    FitIns,
    EvaluateIns,
)

from confg.configuracion import exper_config

logger = logging.getLogger(__name__)
logging.getLogger("flwr").propagate = False

# Paths for saving aggregated weights and metrics
pesos_path = Path(exper_config["guardado_pesos"])
metrics_csv = Path(exper_config["metrics_log_path"])
client_metrics_csv = Path(exper_config["client_metrics_log_path"])

# Ensure directories exist
pesos_path.parent.mkdir(parents=True, exist_ok=True)
metrics_csv.parent.mkdir(parents=True, exist_ok=True)
client_metrics_csv.parent.mkdir(parents=True, exist_ok=True)

# Initialize CSVs if they don't exist
if not pesos_path.exists():
    pd.DataFrame(
        columns=[
            "ronda",
            "client_id",
            "Slice",
            "pesos_previos",
            "pesos_nuevos",
            "diferencia_pesos",
        ]
    ).to_csv(pesos_path, index=False)

if metrics_csv.exists():
    metrics_csv.unlink()
if client_metrics_csv.exists():
    client_metrics_csv.unlink()


def flatten_parameters(parameters: fl.common.Parameters) -> np.ndarray:
    """Concatena todos los parámetros en un único vector 1-D."""
    ndarrays = parameters_to_ndarrays(parameters)
    return np.concatenate([x.flatten() for x in ndarrays], axis=0)


def arrays_to_dict(arrays: List[np.ndarray]) -> dict:
    """Convierte una lista de arrays en un dict layer_i -> list."""
    return {f"layer_{i}": arr.tolist() for i, arr in enumerate(arrays)}


def aggregate_client_info() -> List[dict]:
    """
    Lee los CSV de cada cliente para obtener sus propiedades.
    Usa la misma carpeta que en el experimento binario.
    """
    client_dir = Path(exper_config["client_info_dir"])
    files = client_dir.glob("client_*_info.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.to_dict("records")


class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Estrategia FedAvg personalizada para clasificación multiclase (7 clases):
    - Alterna Slice en fit/evaluate por ronda
    - Guarda cambios de pesos en CSV
    - Agrega métricas macro (accuracy, precision, recall, f1)
    - Integra W&B si se desea
    """

    def __init__(
            self,
            *args,
            run_config: dict,
            use_wandb: bool = False,
            project_name: str = "flower-federated",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.previous_parameters = None
        self.use_wandb = use_wandb
        self.run_config = run_config

        if self.use_wandb:
            wandb.init(
                project=project_name,
                config=self.run_config,
                name=f"{project_name}-run",
                reinit=True,
            )

    def configure_evaluate(self, server_round, parameters, client_manager):
        client_manager.wait_for(
            num_clients=exper_config["num_clients"],
            timeout=30,
        )

        if exper_config.get("clientes_random", False):
            all_clients = list(client_manager.all().values())
            num_to_select = exper_config["num_clients"] // 2
            eval_clients = random.sample(all_clients, num_to_select)
            logger.info(
                f"Ronda {server_round}: evaluación aleatoria de {len(eval_clients)} clientes"
            )
        else:
            target_slice = 1 if server_round % 2 == 0 else 0
            eval_clients = []
            for cid, client in client_manager.all().items():
                try:
                    props = client.get_properties(
                        GetPropertiesIns(config={}),
                        timeout=30,
                        group_id=None,
                    ).properties
                    if props.get("Slice", -1) == target_slice:
                        eval_clients.append(client)
                except Exception:
                    continue
            logger.info(
                f"Ronda {server_round}: evaluando {len(eval_clients)} clientes (Slice={target_slice})"
            )

        return [
            (client, EvaluateIns(parameters, {}))
            for client in eval_clients
        ]

    def configure_fit(self, server_round, parameters, client_manager):
        client_manager.wait_for(
            num_clients=exper_config["num_clients"],
            timeout=30,
        )

        # ¿Selección aleatoria o por slice?
        if exper_config.get("clientes_random", False):
            # Tomamos la mitad de todos los clientes de forma aleatoria
            all_clients = list(client_manager.all().values())
            num_to_select = exper_config["num_clients"] // 2
            selected = random.sample(all_clients, num_to_select)
            logger.info(
                f"Ronda {server_round}: selección aleatoria de {len(selected)} clientes"
            )
        else:
            # Lógica actual: slice par/impar
            target_slice = 1 if server_round % 2 == 0 else 0
            selected = []
            for cid, client in client_manager.all().items():
                try:
                    props = client.get_properties(
                        GetPropertiesIns(config={}),
                        timeout=30, group_id=None
                    ).properties
                    if props.get("Slice", -1) == target_slice:
                        selected.append(client)
                except Exception as e:
                    logger.error(f"Error get_properties cliente {cid}: {e}")
            logger.info(
                f"Ronda {server_round}: seleccionados {len(selected)} clientes (Slice={target_slice})"
            )

        # Siempre devolvemos la mitad de clientes
        return [(c, FitIns(parameters, {})) for c in selected]

    def aggregate_fit(self, server_round, results, failures):
        # Llamada base para agregación de parámetros
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Mapeo client_id -> Slice
            client_info = {str(d["client_id"]): d["Slice"] for d in aggregate_client_info()}
            round_records = []
            prev_flat = (
                flatten_parameters(self.previous_parameters) if self.previous_parameters is not None else None
            )

            for _, fit_res in results:
                cid = fit_res.metrics.get("client_id", "Unknown")
                slice_label = client_info.get(str(cid), "Unknown")
                new_arrays = parameters_to_ndarrays(fit_res.parameters)
                new_flat = flatten_parameters(fit_res.parameters)
                delta = (new_flat - prev_flat).tolist() if prev_flat is not None else []
                round_records.append({
                    "ronda": server_round,
                    "client_id": cid,
                    "Slice": slice_label,
                    "pesos_previos": json.dumps(
                        arrays_to_dict(parameters_to_ndarrays(self.previous_parameters))
                    ) if prev_flat is not None else "[]",
                    "pesos_nuevos": json.dumps(arrays_to_dict(new_arrays)),
                    "diferencia_pesos": json.dumps(delta),
                })

            pd.DataFrame(round_records).to_csv(
                pesos_path, mode="a", header=False, index=False
            )
            logger.info(f"Guardados pesos ronda {server_round} para {len(round_records)} clientes en {pesos_path}")

        self.previous_parameters = aggregated_params
        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        # Si no hay resultados, devolvemos vacío
        if not results:
            return None, {}

        # 1) Obtener loss agregado del método base
        loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # 2) Calcular métricas macro ponderadas por num_examples
        total_examples = sum(res.num_examples for _, res in results)
        agg_metrics = {}
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            weighted = sum(
                res.metrics.get(key, 0.0) * res.num_examples for _, res in results
            )
            agg_metrics[key] = (weighted / total_examples) if total_examples else 0.0

        # 3) Guardar métricas por cliente
        per_client = []
        for _, res in results:
            row = {"ronda": server_round}
            row.update(res.metrics)
            per_client.append(row)
        if per_client:
            pd.DataFrame(per_client).to_csv(
                client_metrics_csv,
                mode="a",
                header=not client_metrics_csv.exists(),
                index=False,
            )

        # 4) Guardar métricas agregadas por ronda
        agg_row = {
            "ronda": server_round,
            "loss": float(loss),
            **{k: float(v) for k, v in agg_metrics.items()},
        }
        pd.DataFrame([agg_row]).to_csv(
            metrics_csv,
            mode="a",
            header=not metrics_csv.exists(),
            index=False,
        )

        # 5) Logging
        logger.info(
            f"Ronda {server_round} – Loss: {loss:.4f}, "
            f"Acc: {agg_metrics['accuracy']:.4f}, "
            f"Prec_macro: {agg_metrics['precision_macro']:.4f}, "
            f"Rec_macro: {agg_metrics['recall_macro']:.4f}, "
            f"F1_macro: {agg_metrics['f1_macro']:.4f}"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "eval/loss": loss,
                    **{f"eval/{k}": v for k, v in agg_metrics.items()}
                },
                step=server_round,
            )

        # 6) Devolver loss y métricas
        return float(loss), agg_metrics
