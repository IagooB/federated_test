import os
os.environ["MODO_BINARIO"] = "True"   # o "False" para multiclase
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"

from subprocess import call
call(["flwr", "run", ".", "--run-config", "use-wandb=true"])
