from typing import OrderedDict
import torch
from collections import OrderedDict as odict

class Mask():
    def __init__(self, model:torch.nn.Module, p:float, names_to_prune=None):
        self.model = model
        self.p = p
        if names_to_prune is None:
            names_to_prune = [n for n,_ in model.named_parameters()]
        self.names_to_prune = names_to_prune
        self.mask = None
    
    def get_params_to_prune(self):
        params_to_prune = odict()
        for n, param in self.model.named_parameters():
            if n in self.names_to_prune:
                params_to_prune[n] = param
        return params_to_prune

    def build_mask(self, criterion="least_magnitude"):
        if criterion == "least_magnitude":
            params_to_prune = torch.cat(self.get_params_to_prune().values(), dim=0).sort().values
            quantile = params_to_prune[int(self.p * len(params_to_prune))]
            self.mask = {n: (m.abs() >= quantile) for n, m in self.model.named_parameters() if n in self.names_to_prune}
        else:
            raise AttributeError(f"Unrecognized criterion for mask building: {criterion}")
    
    def apply_mask(self, to_grad=False):
        if self.mask is None:
            raise RuntimeError("Empty mask cannot be applied to the model")
        for (mask_name, mask_values), (par_name, par_values) in zip(self.mask.items(), self.model.named_parameters()):
            if mask_name == par_name:
                if to_grad:
                    par_values.grad.mul_(mask_values)
                else:
                    par_values.data.mul_(mask_values)
            else:
                raise RuntimeError(f"Incompatible names found: model parameters => {par_name} -- mask parameters => {mask_name}")
    
    def get_sparsity(self):
        num_ones = sum([m.sum().item() for m in self.mask.values()])
        num_params = sum([m.numel() for m in self.mask.values()])
        return 1 - num_ones / num_params
    
    def to_(self, torch_device):
        self.mask = {n: m.to(torch_device) for n, m in self.mask.items()}
            
        

