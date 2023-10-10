import torch

class Trainer():
    def __init__(self, model, dataloader, lr=1e-3, foreach:bool=True, batch=64, epochs=10):
       self.lr = lr
       self.batch = batch
       self.epochs = epochs
       self.model = model
       if not model:
           self.fail('model is nil')

       self.dataloader = dataloader
       if not dataloader:
            self.fail("dataloader is nil")

       self.loss_fn = torch.nn.MSELoss ()
       self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, foreach=foreach)

    def StartLoopTrain(self, dir_in:str="model.pth", dir_out:str="model.pth", load_weights:bool=False):
        if load_weights:
            print("loading model weights...")
            self.model =  torch.load(dir_in)

        self.model.train()

        size = len(self.dataloader.dataset)
        for epoch in range (self.epochs):
            for self.batch, (data, label) in enumerate(self.dataloader):
                pred = self.model(data)
                loss = self.loss_fn(pred, data)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.batch % 100 == 0:
                    loss, current = loss.item(),(self.batch+1) * len(data)
                    print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}] ({current/size*100:.2f}% - epoch:{epoch:>5d})")
                    torch.save(self.model, dir_out)
