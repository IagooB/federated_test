import json
import logging
import os
from random import sample
from typing import Tuple, List
from pathlib import Path
import flwr as fl
import numpy as np
import pandas as pd
import wandb
from flwr.common import EvaluateIns
from flwr.common import parameters_to_ndarrays, GetPropertiesIns, FitIns

from confg.configuracion import *

logger = logging.getLogger(__name__)
logging.getLogger("flwr").propagate = False

pesos_path = exper_config["guardado_pesos"]

if pesos_path.exists():
    pesos_path.unlink()

pesos_path.parent.mkdir(parents=True, exist_ok=True)

if not pesos_path.exists():
    pd.DataFrame(
        columns=[
            "round",
            "client",
            "slice",
            "initial_weights",
            "updated_weights",
            "delta_weights",
            "delta_flat",
        ]
    ).to_csv(pesos_path, index=False)


def flatten_parameters(parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    return np.concatenate(ndarrays, axis=None)


def arrays_to_dict(arrays):
    """Convierte una lista de arrays en un diccionario con claves únicas."""
    return {f"layer_{i}": array.tolist() for i, array in enumerate(arrays)}


def aggregate_client_info():
    """Aggregate client information from CSV files."""
    client_files = Path(exper_config["client_info_dir"]).glob("client_*_info.csv")
    all_data = pd.concat([pd.read_csv(f) for f in client_files], ignore_index=True)
    return all_data.to_dict("records")


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(
            self,
            *args,
            run_config: dict,
            use_wandb: bool = False,
            project_name: str = "flower-federated",
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb
        self.run_config = run_config

        if self.use_wandb:
            wandb.init(
                project=project_name,
                config=self.run_config,
                name=f"{project_name}-run",
                reinit=True,
            )

        self.previous_parameters = None

        # Paths for metric logs (unchanged)
        self.metrics_csv = Path(exper_config.get("metrics_log_path"))
        self.client_metrics_csv = Path(exper_config.get("client_metrics_log_path"))
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        self.client_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        if self.metrics_csv.exists():
            self.metrics_csv.unlink()
        if self.client_metrics_csv.exists():
            self.client_metrics_csv.unlink()

    def configure_evaluate(
            self,
            server_round: int,
            parameters: fl.common.Parameters,
            client_manager: fl.server.ClientManager
    ):
        # Esperamos a que TODOS los clientes se registren
        client_manager.wait_for(
            num_clients=exper_config["num_clients"],
            timeout=30,
        )

        # Alternamos slice por ronda (igual que en configure_fit)
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

        # Sólo esos clientes ejecutarán evaluate()
        return [
            (client, EvaluateIns(parameters, {}))
            for client in eval_clients
        ]

    def configure_fit(
            self,
            server_round: int,
            parameters: fl.common.Parameters,
            client_manager: fl.server.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        """
        Selecciona clientes para la ronda de entrenamiento.
        - Si clientes_random=True: toma una muestra aleatoria de clientes mixtos.
        - Si clientes_random=False: alterna Slice fijo por ronda.
        """
        # Esperar registro de todos los clientes
        client_manager.wait_for(
            num_clients=exper_config["num_clients"],
            timeout=30,
        )

        # Lista de todos los proxies de cliente
        all_clients = list(client_manager.all().values())

        # Flag de muestreo aleatorio puro
        if exper_config["clientes_random"]:
            # Selección aleatoria de clientes mixtos (mezcla slices)
            selected = sample(
                all_clients,
                exper_config["clients_por_ronda"]
            )
        else:
            # Selección fija alternando slice por ronda
            target_slice = 1 if server_round % 2 == 0 else 0
            selected = []
            for client in all_clients:
                try:
                    props = client.get_properties(
                        GetPropertiesIns(config={}),
                        timeout=30,
                        group_id=None,
                    ).properties
                    if props.get("Slice", -1) == target_slice:
                        selected.append(client)
                except Exception as e:
                    logger.error(
                        f"Error obteniendo propiedades del cliente durante configure_fit: {e}"
                    )
                    continue

        logger.info(
            f"Ronda {server_round}: seleccionados {len(selected)} clientes"
            f" (clientes_random={exper_config['clientes_random']}, "
            f"clients_por_ronda={exper_config['clients_por_ronda']})"
        )

        # Devolver lista de (cliente, instrucciones de fit)
        return [
            (client, FitIns(parameters, {}))
            for client in selected
        ]

    def aggregate_fit(
            self,
            server_round: int,
            results,
            failures,
    ):
        # Run base aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Map client_id to slice
            client_info = {str(i["client_id"]): i["Slice"] for i in aggregate_client_info()}
            records = []

            # Prepare previous arrays and flat
            prev_params = self.previous_parameters
            prev_arrays = parameters_to_ndarrays(prev_params) if prev_params is not None else None
            prev_flat = (flatten_parameters(prev_params) if prev_params is not None else None)

            for client_proxy, fit_res in results:
                cid = fit_res.metrics.get("client_id", "unknown")
                slice_label = client_info.get(str(cid), "unknown")

                # Current arrays and flat
                curr_arrays = parameters_to_ndarrays(fit_res.parameters)
                curr_flat = flatten_parameters(fit_res.parameters)

                # initial_weights
                initial_dict = arrays_to_dict(prev_arrays) if prev_arrays is not None else {}
                # updated_weights
                updated_dict = arrays_to_dict(curr_arrays)

                # delta_weights as dict of arrays
                if prev_arrays is not None:
                    delta_arrays = [c - p for c, p in zip(curr_arrays, prev_arrays)]
                    delta_dict = arrays_to_dict(delta_arrays)
                    delta_flat_list = (curr_flat - prev_flat).tolist()
                else:
                    delta_dict = {}
                    delta_flat_list = []

                records.append({
                    "round": server_round,
                    "client": cid,
                    "slice": slice_label,
                    "initial_weights": json.dumps(initial_dict),
                    "updated_weights": json.dumps(updated_dict),
                    "delta_weights": json.dumps(delta_dict),
                    "delta_flat": json.dumps(delta_flat_list),
                })

            # Append to CSV
            pd.DataFrame(records).to_csv(
                pesos_path,
                mode="a",
                header=False,
                index=False,
            )
            logger.info(
                f"Guardados datos de ronda {server_round} para {len(records)} clientes en {pesos_path}"
            )

            # Update previous parameters
            self.previous_parameters = aggregated_params

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        # 0) Saltar si no hay resultados
        if not results:
            return None, {}

        # 1) Calcular la accuracy agregada manualmente
        #    Sumamos accuracy * num_examples de cada cliente
        accuracies = [
            eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ]
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        aggregated_accuracy = (sum(accuracies) / total_examples) if total_examples else 0.0

        # 2) Volcar métricas por cliente a CSV
        per_client = []
        for _, eval_res in results:
            row = {"ronda": server_round}
            row.update(eval_res.metrics)  # debe contener client_id, accuracy, precision_label_X…
            per_client.append(row)
        if per_client:
            df_pc = pd.DataFrame(per_client)
            df_pc.to_csv(
                self.client_metrics_csv,
                mode="a",
                header=not self.client_metrics_csv.exists(),
                index=False,
            )

        # 3) Delegar el cálculo de loss y resto de métricas a la estrategia base
        loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 4) Volcar métricas agregadas por ronda (incluyendo la accuracy calculada)
        agg_row = {
            "ronda": server_round,
            "loss": float(loss),
            "accuracy": float(aggregated_accuracy),
            **{k: float(v) for k, v in aggregated_metrics.items()}
        }
        df_agg = pd.DataFrame([agg_row])
        df_agg.to_csv(
            self.metrics_csv,
            mode="a",
            header=not self.metrics_csv.exists(),
            index=False,
        )

        # 5) Log único con loss, accuracy y métricas adicionales
        logger.info(
            f"Ronda {server_round} – "
            f"Loss: {loss:.4f}, "
            f"Accuracy: {aggregated_accuracy:.4f}, "
        )

        if self.use_wandb:
            wandb.log(
                {
                    "eval/loss": loss,
                    "eval/accuracy": aggregated_accuracy,
                    **{f"eval/{k}": v for k, v in aggregated_metrics.items()},
                },
                step=server_round,
            )
        # 6) Devolver loss y un dict que incluye la accuracy como antes
        return float(loss), {"accuracy": float(aggregated_accuracy), **aggregated_metrics}
