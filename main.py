import torch 
from src.neural import arch
from src.neural import train
from src.dataset import dataset

model = arch.Model()
batch = 522

print(model)
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

dataset.batch = batch
trainer = train.Trainer(model, dataset.train_dataloader, batch=batch)
trainer.StartLoopTrain()

