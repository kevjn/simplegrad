import numpy as np
from abc import abstractmethod

class Optimizer:
    # TODO: fix decay
    def __init__(self, params, *args, decay=0.):
        self.params = params
        self.args = args
        self.t = 0

    def step(self):
        self.t += 1
        self._step(self.t, *self.args)

    @abstractmethod
    def _step(self):
        raise NotImplementedError

    def zero_grad(self):
        for t in self.params:
            t.grad = 0.0

class Adam(Optimizer):
    
    def __init__(self, params, learning_rate = 0.01, epsilon = 1e-7, 
                       beta1 = 0.9, beta2 = 0.999, **kwargs):

        super().__init__(params, *(learning_rate, epsilon, beta1, beta2), **kwargs)

        self.moment = [np.zeros(p.shape) for p in params]
        self.cache = [np.zeros(p.shape) for p in params]

    def _step(self, t, lr, eps, b1, b2):
        # bias correction
        lr *= ((1 - b2**t)**0.5) / (1.0 - b1**t)
        
        for t, m, v in zip(self.params, self.moment, self.cache):
            m[:] = b1 * m + (1.0 - b1) * t.grad
            v[:] = b2 * v + (1.0 - b2) * t.grad * t.grad
            t.val -= lr * m / (np.sqrt(v) + eps)
