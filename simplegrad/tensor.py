import functools
import types
import numpy as np
import typing
import itertools as it
import operator

class Tensor(object):

    def __init__(self, value, parent=None):
        self.debug = ""
        self.cache = -1
        self.val = np.array(value)
        self.parent = parent
        self.grad = 0

        self.visited = False

    def __repr__(self):
        return f"Tensor({self.val!r},\ngrad={self.grad!r})"

    def _backward(self):
        # executed last in the backward pass
        if self.parent:
            self.parent.grad += self.grad
        return self.grad

    def backward(self):
        # implicit gradient creation for first grad
        self.grad = np.ones_like(self.val)

        self.visited = True
        return self._backward()
        
    def chain_rule(self, next_backward, backward, inp, operand):
        @functools.wraps(next_backward)
        def wrapper():
            assert len(operand) < 2

            *oper_grad, self.grad = next_backward(self.grad, inp, *operand)

            # propagate back on operands for multiple gradient output
            for oper in operand:
                oper.grad += oper_grad[0]

                if oper.visited:
                    continue # don't loop

                oper.visited = True
                oper._backward()

            return backward()
        return wrapper

    @property
    def shape(self):
        return self.val.shape

    def __getitem__(self, idx):
        return self.val[idx]

    def operation(expr):
        @functools.wraps(expr)
        def wrapper(self, *rhs, **np_kwargs):
            self.debug += ' ' + expr.__name__ + ' '
            assert len(rhs) < 2, "can't use more than 1 operand currently"

            forward, backward = expr(self, **np_kwargs)

            # save intermediate inputs for backward pass
            lhs = self.val

            # use cache if operand is self
            self.val = forward((self.val, self.cache)[self in rhs], *(x.val for x in rhs))

            self.cache = lhs

            self._backward = self.chain_rule(backward, self._backward, lhs, rhs)

            return self

        # if expr.__name__ in operator.__all__:
        #     # override __(expr.__name__)__ method
        return wrapper

    def parse_args(oper):
        @functools.wraps(oper)
        def wrapper(self, *args, **kwargs):
            return oper(self, *self.parse_operand(args), **kwargs)
        return wrapper

    @functools.singledispatchmethod
    def parse_operand(self, args):
        return args.val
        raise NotImplementedError()

    @parse_operand.register
    def _(self, args: tuple):
        return [self.parse_operand(arg) for arg in args]

    @parse_operand.register
    def _(self, args: types.FunctionType):
        return args()
    
    @parse_operand.register
    def _(self, args: np.ndarray):
        return args

    @parse_operand.register
    def _(self, args: float):
        return args

    @parse_operand.register
    def _(self, args: int):
        return Tensor(args)

    # ***** unary ops *****
    @operation
    def relu(self):

        def forward(x):
            return np.maximum(x, 0)

        def backward(dv, x):
            return dv * (x >= 0),
        
        return forward, backward

    # ***** reduce ops *****
    @operation
    def sum(self, **np_kwargs):

        def forward(x):
            return np.sum(x, **np_kwargs)
        
        def backward(dv, x):
            if x.ndim > dv.ndim:
                dv = np.expand_dims(dv, -1)
            return dv + np.zeros_like(x), # use broadcasting to extend array

        return forward, backward

    @operation
    def exp(self): # TODO: remove self

        def forward(x):
            return np.exp(x)

        def backward(dv, x):
            return dv * np.exp(x),

        return forward, backward

    @operation
    def log(self):
        def forward(x):
            # x = np.clip(x, 1e-7, 1 - 1e-7)
            return np.log(x)

        def backward(dv, x):
            return dv / x,

        return forward, backward

    @operation
    def max(self, **kwargs):

        def forward(x):
            return np.max(x, **kwargs)
        
        def backward(dv, x):
            mask = (x == np.max(x, **kwargs))
            div = mask.sum(**kwargs)
            return mask*dv / div,

        return forward, backward

    def unbroadcast(arr, shape):
        if shape < arr.shape:
            fill = shape[-1] if shape else None
            return arr.sum(axis=tuple(idx for idx, (a,b) in enumerate(it.zip_longest(arr.shape, shape, fillvalue=fill)) if a!=b)).reshape(shape)
        return arr

    # ***** binary ops *****
    @operation
    def sub(self):

        def forward(x,y):
            return x-y

        def backward(dv, x, y):
            return Tensor.unbroadcast(-dv, y.val.shape), Tensor.unbroadcast(dv, x.shape)

        return forward, backward

    @operation
    def pow(self):

        def forward(x, y):
            return x ** y

        def backward(dv, x, y):
            return Tensor.unbroadcast(dv * x ** y.val * np.log(x), y.val.shape), \
                   Tensor.unbroadcast(dv * y.val * x ** (y.val-1.0), x.shape)

        return forward, backward

    @operation
    def mul(self):

        def forward(x, y):
            return x * y

        def backward(dv, x, y):
            return Tensor.unbroadcast(dv * x, y.val.shape), Tensor.unbroadcast(dv * y.val, x.shape)
            # return dv, y.out

        return forward, backward

    @operation
    def add(self):
        def forward(x, y):
            return x + y

        def backward(dv, x, y):
            return Tensor.unbroadcast(dv, y.val.shape), Tensor.unbroadcast(dv, x.shape)
            return dv, dv

        return forward, backward

    # ***** processing ops *****
    @operation
    def dot(self):

        def forward(x, y):
            return x @ y
        
        def backward(dv, x, y):
            return x.T @ dv, dv @ y.val.T

        return forward, backward

    # ***** composite ops *****
    def div(self, x):
        assert isinstance(x, type(self))
        return self.mul(x.pow(Tensor(np.array([-1.]))))

    def softmax(self):
        _max = Tensor(self.val, parent=self).max(axis=-1, keepdims=True)
        self.sub(_max).exp()
        s = Tensor(self.val, parent=self).sum(axis=-1, keepdims=True)
        return self.div(s)

    # numericaly stable
    def logsoftmax(self):
        _max = Tensor(self.val, parent=self).max(axis=-1, keepdims=True)
        s = Tensor(self.val, parent=self).sub(_max).exp().sum(axis=-1, keepdims=True).log()
        _max.add(s)
        return self.sub(_max)

    def mean(self):
        a = Tensor(self.shape[-1])
        return self.sum(axis=-1, keepdims=True).div(a)