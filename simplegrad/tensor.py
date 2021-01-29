import functools
import types
import numpy as np
import typing
import itertools as it
import operator

class Tensor(object):

    def __init__(self, value):
        self.debug = ""
        self.val = np.array(value)
        self.grad = 0

        self.backward_fxns = [] 
        self.arguments = []

    def __repr__(self):
        return f"Tensor({self.val!r},\ngrad={self.grad!r})"

    @property
    def shape(self):
        return self.val.shape

    def __getitem__(self, idx):
        return self.val[idx]

    def _backward(self, dv_fork, dv_parent, parent):
        parent.grad = dv_parent
        self.grad += dv_fork
        self.grad = self.grad,
        while self.backward_fxns:
            backward = self.backward_fxns.pop()
            args = self.grad + self.arguments.pop()
            self.grad = backward(*args)

        assert len(self.grad) == 1
        self.grad = self.grad[0]

        return parent.grad,

    def backward(self):
        # implicit gradient creation
        self._backward(0, np.ones(self.shape), self)
        
    def operation(expr, type=None):
        assert type, "operation type not specified"
        return type(expr)

        # if expr.__name__ in operator.__all__:
        #     # override __(expr.__name__)__ method
        # TODO: GPU accelerator

    def unary_operation(expr):
        # @functools.wraps(expr)
        def wrapper(self, **np_kwargs):
            self.debug += ' ' + expr.__name__ + ' '

            forward, backward = expr(**np_kwargs)
            args = self.val,

            # save intermediate variables for backward pass
            self.arguments.append(args)
            self.backward_fxns.append(backward)

            self.val = forward(*args)

            return self
        return wrapper

    def binary_operation(expr):
        def wrapper(self, operand, **np_kwargs):
            self.debug += ' ' + expr.__name__ + ' '

            forward, backward = expr(**np_kwargs)

            # unbroadcast results from backward pass
            backward = Tensor.unbroadcast(backward)
            args = (self.val, operand.val)

            # propagate back on operand with parent=self as argument
            self.arguments.append((self,))
            self.backward_fxns.append(operand._backward)

            # save intermediate variables for backward pass
            self.arguments.append(args)
            self.backward_fxns.append(backward) 

            self.val = forward(*args)

            return self
        return wrapper

    operation.unary = functools.partial(operation, type=unary_operation)
    operation.binary = functools.partial(operation, type=binary_operation)

    @staticmethod
    def unbroadcast(df):
        @functools.wraps(df)
        def wrapper(dv, x, y):
            def generator():
                for grad, shape in zip(df(dv, x, y), (y.shape, x.shape)):
                    if len(shape) > 3:
                        yield grad # TODO: Fix for 4d tensors
                        continue
                    if shape < grad.shape:
                        fill = shape[-1] if shape else None
                        yield grad.sum(axis=tuple(idx for idx, (a,b) in \
                            enumerate(it.zip_longest(grad.shape, shape, fillvalue=fill)) if a!=b)).reshape(shape)
                    else:
                        yield grad
            
            return tuple(generator())
        return wrapper

    # ========== unary ops ==========

    @operation.unary
    def relu():

        def forward(x):
            return np.maximum(x, 0)

        def backward(dv, x):
            return dv * (x >= 0),
        
        return forward, backward

    # ========== reduce ops ==========

    @operation.unary
    def sum(**np_kwargs):

        def forward(x):
            return np.sum(x, **np_kwargs)
        
        def backward(dv, x):
            if x.ndim > dv.ndim:
                dv = np.expand_dims(dv, -1)
            return dv + np.zeros_like(x), # use broadcasting to extend array

        return forward, backward
    
    @operation.unary
    def exp():

        def forward(x):
            return np.exp(x)

        def backward(dv, x):
            return dv * np.exp(x),

        return forward, backward

    @operation.unary
    def log():
        def forward(x):
            # x = np.clip(x, 1e-7, 1 - 1e-7)
            return np.log(x)

        def backward(dv, x):
            return dv / x,

        return forward, backward

    @operation.unary
    def max(**kwargs):

        def forward(x):
            return np.max(x, **kwargs)
        
        def backward(dv, x):
            mask = (x == np.max(x, **kwargs))
            div = mask.sum(**kwargs)
            return mask*dv / div,

        return forward, backward

    # ========== binary ops ==========

    @operation.binary
    def sub():

        def forward(x,y):
            return x-y

        def backward(dv, x, y):
            return -dv, dv

        return forward, backward

    @operation.binary
    def pow():

        def forward(x, y):
            return x ** y

        def backward(dv, x, y):
            return dv * x ** y * np.log(x), \
                   dv * y * x ** (y-1.0)

        return forward, backward

    @operation.binary
    def mul():

        def forward(x, y):
            return x * y

        def backward(dv, x, y):
            return dv * x, dv * y

        return forward, backward

    @operation.binary
    def add():
        def forward(x, y):
            return x + y

        def backward(dv, x, y):
            return dv, dv

        return forward, backward

    # ========== processing ops ==========

    @operation.binary
    def dot():

        def forward(x, y):
            return x @ y
        
        def backward(dv, x, y):
            return x.T @ dv, dv @ y.T

        return forward, backward

    @operation.binary
    def conv2d(padding=0, strides=1): # kernel size
        # TODO: fix strides
        def forward(x, w):
            assert x.ndim, w.ndim == (4, 4)
            # cross correlation
            # w = np.flipud(np.fliplr(w))

            N, cin, Hin, Win = x.shape
            cout, cin, ky, kx = w.shape

            Hout = Hin - ky + 1
            Wout = Win - kx + 1

            out_strides = (x.strides[0],) + x.strides[2:] + x.strides[1:]

            out_shape = (N, Hout, Wout, cin, ky, kx)

            x = np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)

            # swap axis to mimic pytorch
            return np.tensordot(w, x, axes=((1,2,3), (-3,-2,-1))).swapaxes(0,1)

        def backward(dv, x, w):
            dx = forward(x.swapaxes(0,1), dv.swapaxes(0,1))

            # pad with zeros to simulate a 'full' convolution
            padding = ((0,0),) * 2 + tuple(map(lambda x: (x//2,)*2, dv.shape[-2:]))
            w = np.pad(w, pad_width=padding, mode='constant')

            dw = forward(w.swapaxes(0,1), dv).swapaxes(0,1)
            return dx, dw

        return forward, backward

    # ========== composite ops ==========

    def div(self, x):
        assert isinstance(x, type(self))
        return self.mul(x.pow(Tensor(-1.0)))

    def softmax(self):
        _max = self.fork().max(axis=-1, keepdims=True)
        self.sub(_max).exp()
        s = self.fork().sum(axis=-1, keepdims=True)
        return self.div(s)

    def logsoftmax(self):
        _max = self.fork().max(axis=-1, keepdims=True)
        _sub = self.fork().sub(_max).exp().sum(axis=-1, keepdims=True).log()
        _max.add(_sub)
        return self.sub(_max)

    def mean(self):
        a = Tensor(self.shape[-1])
        return self.sum(axis=-1, keepdims=True).div(a)

    # ========== control flow ops ==========

    def fork(self):
        self.debug += " fork "

        def fork_backward(dv):
            self.grad += dv # accumulate parent gradient
            return dv, 

        fork = Tensor(self.val)
        fork.backward_fxns.append(fork_backward)
        fork.arguments.append(tuple()) # empty args

        return fork