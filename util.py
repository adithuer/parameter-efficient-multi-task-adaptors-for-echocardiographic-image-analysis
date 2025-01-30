import re
import wandb

from enum import Enum
from tqdm import tqdm

import torch
from torchmetrics import Accuracy, MeanMetric, Dice, Recall, Precision, MeanSquaredError, MeanAbsoluteError



class TrainMethod(Enum):
    FINETUNE = 'finetune'
    PRETRAIN = 'pretrain'
    FULL = 'full'
    LORA = 'lora'

class TrainMode(Enum):
    TRAIN = 'TRAIN'
    VALIDATION = 'VAL'
    TEST = 'TEST'

class ModelTasks(Enum):
    SEGMENTATION = 'seg'
    REGRESSION = 'reg'
    CLASSIFICATION = 'clf'

class PretrainedModelMode(Enum):
    FULL = 'full'
    BACKBONE = 'backbone'

class Metric():
    def __init__(self, device: torch.device, n_classes: int, task: str = None) -> None:
        self.task = task
        self.mean = MeanMetric().to(device)      
        self.recall = Recall(task="multiclass", num_classes=n_classes).to(device)
        self.precision = Precision(task="multiclass", num_classes=n_classes).to(device)

        if self.task == ModelTasks.CLASSIFICATION.value:
            self.accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)
        elif self.task == ModelTasks.SEGMENTATION.value:
            self.dice = Dice(num_classes=n_classes, ignore_index=0).to(device)
        elif self.task == ModelTasks.REGRESSION.value:
            self.mse = MeanSquaredError(squared=False).to(device)
            self.rmse = MeanSquaredError(squared=True).to(device)
            self.mae = MeanAbsoluteError().to(device)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        pred_arg = torch.argmax(predictions, dim=1)
        targets_arg = torch.argmax(targets, dim=1)        
        self.mean(loss)
        self.recall(pred_arg, targets_arg)
        self.precision(pred_arg, targets_arg)

        if self.task == ModelTasks.CLASSIFICATION.value:
            self.accuracy(pred_arg, targets_arg)
        elif self.task == ModelTasks.SEGMENTATION.value:
            self.dice(pred_arg, targets_arg)
        elif self.task == ModelTasks.REGRESSION.value:
            self.mse(predictions, targets)
            self.rmse(predictions, targets)
            self.mae(predictions, targets)

    def reset(self) -> None:
        self.mean.reset()  
        self.recall.reset()
        self.precision.reset()

        if self.task == ModelTasks.CLASSIFICATION.value:
            self.accuracy.reset()
        elif self.task == ModelTasks.SEGMENTATION.value:
            self.dice.reset()
        elif self.task == ModelTasks.REGRESSION.value:
            self.mse.reset()
            self.rmse.reset()
            self.mae.reset()

    def get_metrics(self) -> dict:
        results =  {
            'loss':self.mean.compute().item(),
            'recall': self.recall.compute().item(), 
            'precision': self.precision.compute().item()
        }
    
        if self.task == ModelTasks.CLASSIFICATION.value:
            results = {**results, 'accuracy':self.accuracy.compute().item()}
        elif self.task == ModelTasks.SEGMENTATION.value:
            results = {**results, 'dice':self.dice.compute().item()}
        elif self.task == ModelTasks.REGRESSION.value:
            results = {**results, 
                       'mse':self.mse.compute().item(),
                       'rmse':self.rmse.compute().item(),
                       'mae':self.mae.compute().item()}

        return results
    
def train_one_epoch(epoch: int, model, data_loader, optimizer, criterion, device: torch.device, metric):
    metric.reset()
    model.train()

    for input, target in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
        input = input.to(device)
        target = target.to(device)

        pred = model(input)
        optimizer.zero_grad()
        loss = criterion(pred, target)
        metric(pred, target, loss)
        loss.backward()
        optimizer.step()

    return metric.get_metrics()


def validate_one_epoch(epoch: int, model, data_loader, criterion, device: torch.device, metric):
    metric.reset()
    model.eval()

    with torch.no_grad():
        for input, target in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
            input = input.to(device)
            target = target.to(device)
            pred = model(input)            
            loss = criterion(pred, target)            
            metric(pred, target, loss)

    return metric.get_metrics()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def get_config_name(filename: str) -> str:
    return re.sub(r"^config_", '', filename)


def load_model(model, id, wandb_run: wandb=None, project: str=None):
    if wandb_run:
        project = wandb_run._project

    try:
        path = project + '/runs/' + id
        model_weights = wandb.restore(f'model_{id}.pth', run_path=path)
    except:
        raise Exception('Model weights not found')
    
    model.load_state_dict(torch.load(model_weights.name, weights_only=True))
    return model


class EarlyStopper():
    def __init__(self, patience: int = 10, delta: float = 1e-3) -> None:
        self.last_min_val_loss = float("inf")
        self.last_max_val_dice = 0
        self.counter_loss = 0
        self.counter_dice = 0
        self.patience = patience
        self.delta = delta

    def __call__(self, val_loss: float, val_dice: float=None) -> bool:
        if self.last_min_val_loss > val_loss and not abs(self.last_min_val_loss - val_loss) < self.delta:
            self.counter_loss = 0
            self.last_min_val_loss = val_loss
        else:
            self.counter_loss += 1

        if val_dice:
            if self.last_max_val_dice < val_dice and not abs(self.last_max_val_dice - val_dice) < self.delta:
                self.counter_dice = 0
                self.last_max_val_dice = val_dice
            else:
                self.counter_dice += 1

            if self.counter_dice >= self.patience:
                return True
            
        if self.counter_loss >= self.patience:
            return True
        else:
            return False
