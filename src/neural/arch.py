import torch

class Model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.activation = torch.nn.Tahn()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.activation,
                nn.Linear(512, 512),
                nn.activation,
                nn.Linear(512,10),
                )

    def forward(self, input):
        input = self.flatten(input)
        logits = self.layers(input)
        return logits
