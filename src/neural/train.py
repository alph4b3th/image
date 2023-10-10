import torch

class Trainer():
    def __init__(self, model, dataloader, lr=1e-3, batch=64, epochs=10):
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
       self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

    def StartLoopTrain(self):
        size = len(self.dataloader.dataset)
        self.model.train()
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
