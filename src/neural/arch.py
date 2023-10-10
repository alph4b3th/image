import torch


class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.activation = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(28*28, 3028),
                self.activation,
                torch.nn.Linear(3028, 3028),
                self.activation,
                torch.nn.Linear(3028, 512),
                self.activation,
                torch.nn.Linear(512,28*28),
                )
        self.final_shape = torch.nn.Sequential(
                torch.nn.Sigmoid(),
                Reshape(-1, 1, 28, 28)
                )

    def forward(self, input):
        input = self.flatten(input)
        logits = self.layers(input)
        logits = self.final_shape(logits)
        logits = self.activation(logits)
        return logits
