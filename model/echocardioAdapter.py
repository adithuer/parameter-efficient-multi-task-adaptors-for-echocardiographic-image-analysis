import torch
import torch.nn as nn

import copy
from peft import LoraConfig, get_peft_model
from typing import List, Union, Optional

from util import PretrainedModelMode

class EchocardioAdapter(nn.Module):
    def __init__(
        self,
        model_pretrained: nn.Module,
        rank: int,
        target_modules: Optional[Union[List[str], str]],
        trainable_modules: List[str],
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        adapterHead: nn.Module = None,
        model_mode: str = PretrainedModelMode.FULL.value,
    ) -> None:
        super(EchocardioAdapter, self).__init__()

        if model_mode == PretrainedModelMode.FULL.value:
            backbone = copy.deepcopy(model_pretrained)

            if target_modules == 'all':
                target_modules_encoder = [n for n, m in model_pretrained.encoder.named_modules() if self.is_lora_type(m)]
                target_modules_neck = [n for n, m in model_pretrained.neck.named_modules() if self.is_lora_type(m)]
                target_modules_head = [n for n, m in model_pretrained.head.named_modules() if self.is_lora_type(m)]
                target_modules = target_modules_encoder + target_modules_neck + target_modules_head
        else:
            backbone = nn.Sequential(
                copy.deepcopy(model_pretrained.encoder),
                copy.deepcopy(model_pretrained.neck)
            )

            if target_modules == 'all':
                target_modules_encoder = [n for n, m in model_pretrained.encoder.named_modules() if self.is_lora_type(m)]
                target_modules_neck = [n for n, m in model_pretrained.neck.named_modules() if self.is_lora_type(m)]
                target_modules = target_modules_encoder + target_modules_neck

        config = LoraConfig(
                r=rank,
                target_modules=target_modules,
                modules_to_save=trainable_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
        )
        model = nn.Sequential(backbone, adapterHead)        
        self.adapter = get_peft_model(model, config)

        print("trainable weights:")
        self.adapter.print_trainable_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.adapter(x)
        return out

    def is_lora_type(self, module: torch.nn.modules) -> bool:
        if (isinstance(module, torch.nn.modules.linear.Linear) or 
            isinstance(module, torch.nn.modules.conv.Conv2d)):
            return True
        else:
            return False
