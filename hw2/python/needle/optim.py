"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # https://discuss.pytorch.org/t/how-does-sgd-weight-decay-work/33105/2
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            grad = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * grad
            #TODO: figure out if it should be ndl.Tensor(grad, dtype=param.dtype, requires_grad=False)
            self.u[param] = ndl.Tensor(grad, dtype=param.dtype)
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # self.t += 1
        # for w in self.params:
        #     grad = w.grad.data + self.weight_decay * w.data
        #     self.m[w] = self.beta1 * self.m.get(w, 0) + (1 - self.beta1) * grad
        #     self.v[w] = self.beta2 * self.v.get(w, 0) + (1 - self.beta2) * (grad ** 2)
        #     unbiased_m = self.m[w] / (1 - self.beta1 ** self.t)
        #     unbiased_v = self.v[w] / (1 - self.beta2 ** self.t)
        #     w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)
        self.t += 1
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad ** 2)
            m_hat = (self.m[param] / (1 - self.beta1 ** self.t)).detach()
            v_hat = (self.v[param] / (1 - self.beta2 ** self.t)).detach()
            update = ndl.Tensor(self.lr * m_hat / (v_hat ** 0.5 + self.eps), dtype=param.dtype).detach()
            param.data -= update.detach()
        ## END YOUR SOLUTION
