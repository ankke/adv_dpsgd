import torch
import torchvision
import opacus
import copy

import config

import torchvision.transforms as transforms
import torch.optim as optim

from res_net import ResNet9

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

class Model:
    def __init__(self, dataset='cifar'):
        self.dataset = dataset

    def setup(self, epochs, batch_size, dp, target_epsilon, max_grad_norm, device):
        if self.dataset == 'cifar':
            train_set, test_set, self.classes = get_CIFAR()
            in_channels = 3
        elif self.dataset == 'mnist':
            train_set, test_set, self.classes = get_MNIST()
            in_channels = 1
        else:
            raise Exception

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)
        test_loader_x1 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

        self.net = ResNet9(norm_layer="group", in_channels=in_channels)
        self.adv_net = ResNet9(norm_layer="group", in_channels=in_channels)
        optimizer = optim.NAdam(self.net.parameters())

        if dp:
            print("DP on")
            self.privacy_engine = opacus.PrivacyEngine()
            self.net, optimizer, train_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.net,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=target_epsilon,
                target_delta=1e-5,
                max_grad_norm=max_grad_norm,
                noise_generator=torch.Generator(device=device).manual_seed(config.seed)
            )
        else:
            print("DP off")

        return optimizer, train_loader, test_loader, test_loader_x1

    def forward(self, inputs):
        return self.net(inputs)
        
    def to(self, device):
        self.net.to(device)
        self.adv_net.to(device)

    def mode(self, eval):
        if eval:
            self.net.eval()
        else:
            self.net.train()

    def get_model(self, dp):
        if dp:
            self.adv_net.load_state_dict(copy.deepcopy(self.net._module.state_dict()))
            model = self.adv_net
        else:
            model = self.net
        return model
        

def get_CIFAR():
    transform = transforms.Compose(
            [transforms.ToTensor(),
            # transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV) 
            ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_set, test_set, classes

def get_MNIST():
    transform = transforms.Compose(
            [transforms.ToTensor(),
            ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    classes = (x for x in range(0,9))
    return train_set, test_set, classes