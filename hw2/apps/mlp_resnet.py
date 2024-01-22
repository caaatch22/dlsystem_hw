import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
                       norm(dim=hidden_dim),
                       nn.ReLU(),
                       nn.Dropout(p=drop_prob),
                       nn.Linear(in_features=hidden_dim, out_features=dim),
                       norm(dim=dim))   
    
    residual = nn.Residual(fn)
    block = nn.Sequential(residual,
                          nn.ReLU())
    return block
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim),
                           nn.ReLU(),
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes))
    return resnet
    ### END YOUR SOLUTION

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:    
        model.eval()
        for X, y in dataloader:
            logits = model(X.reshape((X.shape[0], -1)))
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            X = X.reshape((X.shape[0], -1))
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error / sample_nums, np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)

    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)

    resnet = MLPResNet(784, hidden_dim)

    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    print("epochs = ", epochs)
    for i in range(epochs):
      print("i = ", i)
      train_acc, train_loss = epoch(dataloader=train_dataloader, model=resnet, opt=opt)
      test_acc, test_loss = epoch(dataloader=test_dataloader, model=resnet)

    return train_acc, train_loss, test_acc, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")