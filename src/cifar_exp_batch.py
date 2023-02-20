import torch
import torchattacks
import random

import config

import numpy as np

from experiment import Experiment

seed = config.seed

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

batch_sizes = [1024]
seeds = [13, 32, 50, 71, 100]
adv_attack_modes =['batch']
dp_mode=[True] 
epsilons = [7.5]
epochs_range = [50]
max_grad_norms = [1.5]
advs = [
    {
        'attack': None, 
        'params': None
    }, 
    # {
    #     'attack': torchattacks.FFGSM, 
    #     'params': {'eps':2/255}
    # }, 
    # {
    #     'attack': torchattacks.PGD, 
    #     'params': {'eps':2/255}
    # }, 
    # {
    #     'attack': torchattacks.PGDL2, 
    #     'params': {'eps':0.1}
    # }, 
    ]

id = 1
for seed in seeds:
    set_seed(seed)
    for batch_size in batch_sizes:
        for dp in dp_mode:
            for adv_attack_mode in adv_attack_modes:
                for epsilon in epsilons:
                    for epochs in epochs_range:
                        for max_grad_norm in max_grad_norms:
                            for adv in advs:
                                try:
                                    Experiment(
                                        batch_size, 
                                        epochs, 
                                        adv["attack"], 
                                        adv_attack_mode, 
                                        adv["params"], 
                                        device, 
                                        id,
                                        save=True, 
                                        dp=dp, 
                                        target_epsilon=epsilon, 
                                        dataset='cifar', 
                                        max_grad_norm=max_grad_norm).run()
                                except Exception as e:
                                    print(e)
                                id += 1
