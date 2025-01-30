import torch
import torch.nn as nn

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import os
import wandb

from util import *


class Trainer:
    def __init__(
        self,
        *,
        n_epochs: int,
        model: nn.Module,
        criterion: torch.nn,
        optimizer: torch.optim,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
        metric: Metric,
        wandb_run: wandb,
        scheduler: torch.optim.lr_scheduler = None,
        pretrained_model: nn.Module = None,
        pretrained_model_id: str = None,
        mode: str = None,
        earlyStopper: EarlyStopper = None,
        eval_metric: str = 'dice'
    ):

        self.device = get_device()
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.metric = metric(self.device)
        self.earlyStopper = earlyStopper
        self.wandb_run = wandb_run
        self.mode = mode
        self.eval_metric = eval_metric

        if pretrained_model_id:
            pretrained_model = load_model(
                pretrained_model, pretrained_model_id, wandb_run
            )

        if self.mode == TrainMethod.LORA.value:
            self.model = model(pretrained_model).to(self.device)
        else:
            self.model = model.to(self.device)
        
        self.optimizer = optimizer(params=self.model.parameters())

        if scheduler:
            self.scheduler = scheduler(optimizer=self.optimizer)
        else:
            self.scheduler = None

    def train(self):
        best_metric = 0

        for epoch in range(self.n_epochs):
            results_train = train_one_epoch(
                epoch,
                self.model,
                self.dataloader_train,
                self.optimizer,
                self.criterion,
                self.device,
                self.metric,
            )
            results_val = validate_one_epoch(
                epoch,
                self.model,
                self.dataloader_val,
                self.criterion,
                self.device,
                self.metric,
            )

            res_train = {"train/" + str(key): val for key, val in results_train.items()}
            res_val = {"val/" + str(key): val for key, val in results_val.items()}

            if self.scheduler:
                res_train = {
                    **res_train,
                    "hyperparameter/lr": self.scheduler.get_last_lr()[0],
                }
                self.scheduler.step()

            self.wandb_run.log({**res_train, **res_val})

            if res_val[f"val/{self.eval_metric}"] > best_metric:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.wandb_run.dir, f"model_{self.wandb_run.id}.pth"),
                )
                best_metric = res_val[f"val/{self.eval_metric}"]

            if self.earlyStopper and self.earlyStopper(
                val_loss=res_val["val/loss"], 
                val_dice=None if not "val/dice" in res_val["val/dice"] else None
            ):
                break

        return self.wandb_run


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb_key = os.getenv("WANDB_KEY")
    wandb.login(key=wandb_key)
    hydra_cfg = HydraConfig.get()
    run = wandb.init(
        project=cfg.wandb.project,
        name=get_config_name(hydra_cfg.job.config_name),
        config={
            **cfg.wandb.config,
            "scheduler": cfg.trainer.scheduler._target_ if 'scheduler' in cfg.trainer else None,
            "dataset": hydra_cfg.runtime.choices.dataset,
            "transform": hydra_cfg.runtime.choices.transform,
        },
    )

    transform_train = hydra.utils.instantiate(cfg.transform.train)
    transform_val = hydra.utils.instantiate(cfg.transform.test)
    transform_target = hydra.utils.instantiate(cfg.transform.target)

    if 'augmentation' in cfg:
        transform = hydra.utils.instantiate(cfg.augmentation)
    else:
        transform = None


    dataset_train = hydra.utils.instantiate(
        cfg.dataset,
        type=TrainMode.TRAIN.value,
        transform=transform,
        transform_data=transform_train,
        transform_target=transform_target,
    )
    dataset_val = hydra.utils.instantiate(
        cfg.dataset,
        type=TrainMode.VALIDATION.value,
        transform_data=transform_val,
        transform_target=transform_target,
    )

    dataloader_train = hydra.utils.instantiate(cfg.dataloader.train, dataset=dataset_train)
    dataloader_val = hydra.utils.instantiate(cfg.dataloader.val, dataset=dataset_val)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        wandb_run=run,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
    )
    run = trainer.train()

    if cfg.upload_model:
        run.save(f'model_{run.id}.pth')
    run.finish()


if __name__ == "__main__":
    main()
