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
        self.act_layer = torch.nn.Tanh()
        self.act_out = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(28*28, 128),
                self.act_layer,
                torch.nn.Linear(128, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 512),
                self.act_layer,
                torch.nn.Linear(512, 2048),
                self.act_layer,
                torch.nn.Linear(2048, 2048),
                self.act_layer,
                torch.nn.Linear(2048, 2048),
                self.act_layer,
                torch.nn.Linear(2048, 2048),
                self.act_layer,
                torch.nn.Linear(2048, 512),
                self.act_layer,
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
        logits = self.act_out(logits)
        return logits
