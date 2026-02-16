from rfdetr import RFDETRNano
from rfdetr import RFDETRSmall
from rfdetr import RFDETRMedium
from rfdetr import RFDETRLarge

import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

tp = params["train"]  # Training params
model_type = params["model"]
dataset = params["dataset"]
output = params["output"]

if str.lower(model_type) == "nano":
    model = RFDETRNano()
elif str.lower(model_type) == "small":
    model = RFDETRSmall()
elif str.lower(model_type) == "medium":
    model = RFDETRMedium()
elif str.lower(model_type) == "large":
    model = RFDETRLarge()

model.train(
    lr = tp["lr"],
    lr_encoder = tp["lr_encoder"],
    batch_size = tp["batch_size"],
    grad_accum_steps = tp["grad_accum_steps"],
    epochs = tp["pochs"],
    dataset_dir = tp["dataset_dir"],
    output_dir = tp["output_dir"],
    weight_decay = tp["weight_decay"],
    tensorboard = tp["tensorboard"],
    wandb = tp["wandb"],
    mlflow = tp["mlflow"],
    clearml = tp["clearml"],
    run_test = tp["run_test"],
    eval_max_dets = tp["max_eval_dets"]
)

