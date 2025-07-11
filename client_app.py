from typing import Dict

from flwr.client import Client, ClientApp
from flwr.common import (
    GetParametersIns,
    GetParametersRes,
    GetPropertiesRes,
    GetPropertiesIns,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Status,
    Code,
)
from sklearn.metrics import precision_score, recall_score, f1_score

from confg.configuracion import *
from logg import logger
from modelos import create_global_model
from task import *
from task import load_data, train, get_weights, set_weights

# Asegurarnos de que exista el directorio 'data' para los CSV de cliente
Path("data").mkdir(exist_ok=True)


class FlowerClient(Client):
    def __init__(self, net: torch.nn.Module, train_loader, test_loader, client_id: int, slice_label: int):
        self.client_id = client_id
        self.slice_label = slice_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo y datos
        self.net = net.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Propiedades que el servidor usa para seleccionar Slice
        self.properties: Dict[str, int] = {
            "client_id": self.client_id,
            "Slice": self.slice_label,
        }

        base = Path(exper_config["client_info_dir"])
        base.mkdir(parents=True, exist_ok=True)

        # CSV de logging de slice por cliente (una sola vez)
        self.csv_path = base / f"client_{self.client_id}_info.csv"
        if not self.csv_path.exists():
            pd.DataFrame(columns=["client_id", "Slice"]).to_csv(self.csv_path, index=False)
        self._log_client_info()

        logger.info(f"[Cliente {self.client_id}] Inicializado (Slice={self.slice_label})")

    def _log_client_info(self) -> None:
        """Escribe client_id y Slice en CSV (sobrescribe)."""
        pd.DataFrame([self.properties]).to_csv(self.csv_path, index=False)
        logger.debug(f"[Cliente {self.client_id}] Info guardada en {self.csv_path}")

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=self.properties,
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        params = get_weights(self.net)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=params,
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Cargar parámetros globales
        set_weights(self.net, ins.parameters)
        # Entrenar local
        train(self.net, self.train_loader, num_epochs=exper_config["local_epochs"])
        # Devolver parámetros actualizados
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=get_weights(self.net),
            num_examples=len(self.train_loader.dataset),
            metrics={"client_id": self.client_id},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        set_weights(self.net, ins.parameters)
        loss, accuracy = evaluate(self.net, self.test_loader)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.net(X_batch.to(self.device))
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        avg_type = "binary" if exper_config["modo_binario"] else "macro"

        precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics=metrics,
        )


def client_fn(context) -> Client:
    """Factory de cliente que Flwr invoca por nodo."""
    pid = context.node_config["partition-id"]
    n_part = context.node_config["num-partitions"]
    train_loader, test_loader, slice_label = load_data(pid, n_part)
    # shape_sinlabel debe seguir indicando sólo la dimensión de entrada
    net = create_global_model(input_shape=input_shape, num_classes=exper_config["num_clases"])
    return FlowerClient(net, train_loader, test_loader, pid, slice_label).to_client()


# Inicializamos la aplicación de cliente de Flower
app = ClientApp(client_fn=client_fn)
