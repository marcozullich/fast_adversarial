import torch

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(yhat, y):
    return (yhat.max(1)[1] == y).sum().item()