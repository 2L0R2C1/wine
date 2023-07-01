import torch.nn as tn

class Neural_network(tn.Module):
    def __init__(self):
        super(Neural_network, self).__init__()
        self.model = tn.Sequential(tn.Linear(13, 32),
                                   tn.Linear(32,64),
                                   tn.ReLU(),
                                   tn.Linear(64,64),
                                   tn.ReLU(),
                                   tn.Linear(64,32),
                                   tn.ReLU(),
                                   tn.Linear(32, 3),
                                   tn.LogSoftmax(dim=1)
                                   )

    def forward(self, x):
        x = self.model(x)
        return x
    