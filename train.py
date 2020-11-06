import torch
import math

def loss_func(prob):
    loss = torch.tensor(len(prob))
    for (i, batch) in enumerate(prob):
        for word in batch:
            loss[i] = loss[i] + math.log(batch[word])
    return loss

def main():
    pass


if __name__ = '__main__':
    main()