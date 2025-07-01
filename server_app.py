from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from confg.configuracion import *
from custom_strategy import CustomFedAvg
from logg import *
from modelos import create_global_model

# Configuración del modelo global

logger.info(f"Servidor iniciado con modelo global de input_shape: {input_shape}")
# Ajustar según los datos reales
net = create_global_model(input_shape, num_classes=exper_config["num_clases"])
init_params = ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])


def server_fn(context):
    strategy = CustomFedAvg(
        initial_parameters=init_params,
        fraction_fit=exper_config["fraction_fit"],
        min_fit_clients=exper_config["clients_por_ronda"],
        min_available_clients=exper_config["num_clients"],
        run_config=context.run_config,
        use_wandb=bool(context.run_config.get("use-wandb", False)),
        project_name=exper_config.get("wandb_project_name", "flower-federated"),
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=exper_config["num_rounds"]))


# Inicializar la aplicación del servidor
app = ServerApp(server_fn=server_fn)
