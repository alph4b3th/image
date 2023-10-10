import torch

class Model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(28*28, 512),
                self.activation,
                torch.nn.Linear(512, 512),
                self.activation,
                torch.nn.Linear(512,10),
                )

    def forward(self, input):
        input = self.flatten(input)
        logits = self.layers(input)
        return logits
