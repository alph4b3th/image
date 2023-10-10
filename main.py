import torch
import argparse
from src.neural import arch
from src.neural import train
from src.dataset import dataset

model = arch.Model()

print(model)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", type=bool, default=False, help="treine o modelo")
parser.add_argument("--epochs", type=int, default=64, help="define os ciclos do treinamento")
parser.add_argument("--batch", type=int, default=64, help="define o batch do dataset")
parser.add_argument("--load-weights", type=bool, help="continua o treinamento de onde parou")
parser.add_argument("--out-weights", type=str, default="./model.pth", help="diretorio para salvar o treinamento")
parser.add_argument("--in-weights", type=str, default="./model.pth", help="diretorio para carregar o treinamento")
parser.add_argument("-l", type=float, default=1e-3, help="taxa de aprendizado, padrão: 1e-3")
parser.add_argument("-f", type=float, default=True, help="foreach")


args = parser.parse_args()
is_train = args.train
epochs = args.epochs
batch = args.batch
load_weights = args.load_weights
dirOut = args.out_weights
dirIn = args.in_weights
lr = args.l
foreach = args.f

print(f"MAIN:{dirOut}")
if is_train:
    dataset.batch = batch
    trainer = train.Trainer(model, dataset.train_dataloader, lr=lr, foreach=foreach, batch=batch, epochs=epochs)
    trainer.StartLoopTrain(dir_in= dirIn, dir_out=dirOut, load_weights=load_weights)


