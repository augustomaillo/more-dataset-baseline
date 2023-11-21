from typing import Callable, List, Union
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

# def warmup(current_step: int):
#     current_step = current_step+1
#     if current_step <= 10:
#         return 3.5e-5*(current_step)
#     elif current_step <= 40:
#         return 3.5e-4
#     elif current_step <= 70:
#         return 3.5e-5
#     elif current_step <= 120:
#         return 3.5e-6
#     else:
#         return 3.5e-7 
    
def warmup(current_step: int):
    current_step = current_step+1
    if current_step <= 10:
        return 3.5e-5
    elif current_step <= 20:
        return 1e-5
    elif current_step <= 70:
        return 3.5e-6
    elif current_step <= 120:
        return 0.5e-6
    return 3.5e-7 
    


class WarmupLR(LambdaLR):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer=optimizer, lr_lambda=warmup)
        