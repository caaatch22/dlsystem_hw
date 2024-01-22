import sys, time
sys.path.append("src/")
from simple_ml import train_softmax, parse_mnist

X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                         "data/train-labels-idx1-ubyte.gz")
X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                         "data/t10k-labels-idx1-ubyte.gz")

start_time = time.time()
train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100)
end_time = time.time()
print("Time elapsed: {:.2f}s".format(end_time - start_time))