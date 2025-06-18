import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

from subprocess import call

call(["flwr", "run", ".", "--run-config", "use-wandb=false"])
