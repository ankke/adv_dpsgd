import torch
import torchattacks
import random

import config

import numpy as np

from experiment import Experiment_weighted

seed = config.seed

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

batch_sizes=[256, 32]
adv_attack_modes =['batch']
dp_mode=[True] 
epsilons = [1.5]
epochs_range = [25]
max_grad_norms = [1.5]
advs = [
    {
        'attack': torchattacks.FFGSM, 
        'params': {'eps':1/255}
    }, 
    {
        'attack': torchattacks.PGD, 
        'params': {'eps':1/255}
    }, 
    {
        'attack': torchattacks.PGDL2, 
        'params': {'eps':0.1}
    }, 
    ]
weights_range = [
    {
        'clean_weight': 99,
        'adv_weight': 1,
    },
    {
        'clean_weight': 9,
        'adv_weight': 1,
    },
    {
        'clean_weight': 5,
        'adv_weight': 5,
    },
]

id = 1

for batch_size in batch_sizes:
    for dp in dp_mode:
        for adv_attack_mode in adv_attack_modes:
            for epsilon in epsilons:
                for epochs in epochs_range:
                    for max_grad_norm in max_grad_norms:
                        for adv in advs:
                            for weights in weights_range:
                                try:
                                    Experiment_weighted(
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
                                        dataset='mnist', 
                                        max_grad_norm=max_grad_norm,
                                        clean_weight=weights["clean_weight"],
                                        adv_weight=weights["adv_weight"],
                                        max_batch_size=512).run()
                                except Exception as e:
                                    print(e)
                                id += 1
