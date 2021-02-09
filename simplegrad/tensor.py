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
        # @functools.wraps(expr)
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


    def typeless_operation(expr):
        def wrapper(self, func, *args, **kwargs):
            nonlocal expr
            # add func to expr closure
            _expr = expr(func.__func__.__closure__[0].cell_contents)
            _expr.__name__ = func.__name__
            # swap wrapper
            func.__func__.__closure__[0].cell_contents = _expr
            # call correct operation type
            func(*args, **kwargs)
            return self
        return wrapper

    operation.unary = functools.partial(operation, type=unary_operation)
    operation.binary = functools.partial(operation, type=binary_operation)
    operation.typeless = functools.partial(operation, type=typeless_operation)

    @staticmethod
    def unbroadcast(df):
        # @functools.wraps(df)
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
            idx = tuple(np.argwhere(x == x.max(**kwargs))[0])
            mask = np.zeros_like(x)
            mask[idx] = 1
            return mask*dv,

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
    def dot(f_subscripts='i...j,j...h->i...h', 
            b_subscripts=('j...i,j...h->i...h','i...j,h...j->i...h')):

        def forward(x, y):
            return np.einsum(f_subscripts,x,y)
            return x @ y
        
        def backward(dv, x, y):
            return *it.starmap(np.einsum, zip(b_subscripts, (x,dv), (dv,y))),
            return x.T @ dv, y.T @ dv

        return forward, backward

    # ========== test ops ==========

    @operation.typeless
    def sliding_window(kernel_fxn): # sliding tensor
        def closure(kernel_size=(2,2), stride=1, **kwargs):
            kernel_forward, kernel_backward = kernel_fxn(**kwargs)
            windows = None
            out = None

            def forward(x, *args):
                nonlocal windows
                nonlocal out

                N, cin, *in_shape = x.shape
                kdims = len(kernel_size)

                truncated_out_shape = *((xin - kin) // 1 + 1 for xin,kin in zip(in_shape, kernel_size)),
                out_shape = N, cin, *truncated_out_shape, *kernel_size
                out_strides = *x.strides[:2], *(xs*stride for xs in x.strides[-kdims:]), *x.strides[-kdims:]

                # generate windows
                windows = np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides)
                ws = windows

                cout = args[0].shape[0] if args else cin

                # init truncated output and set values
                out = np.zeros((N, cout, *truncated_out_shape))
                for N, cout, *i in np.ndindex(*out.shape):
                    out[(N, cout, *i)] = kernel_forward(ws[(N, slice(None), *i)], *(arg[cout] for arg in args))

                return out

            def backward(dv, x, *args):
                nonlocal windows
                nonlocal out
                assert dv.shape == out.shape

                if args:
                    # sparse matrix for each element in dv for binary operations
                    a = np.zeros(out.shape + kernel_size[:1] + windows.shape[-1:])
                    np.einsum('...ii->...i', a)[:] = dv[...,None] 
                    dv = a

                dx = np.zeros_like(x, dtype='float64')
                dw = [np.zeros_like(arg, dtype='float64') for arg in args]

                dx_ws = np.lib.stride_tricks.as_strided(dx, shape=windows.shape, strides=windows.strides)
                x_ws = windows

                for N, cout, *i in np.ndindex(*out.shape):
                    grad = kernel_backward(dv[(N, cout, *i)], \
                            x_ws[(N, slice(None), *i)], *(a[cout] for a in args))
                    # accumulate gradients
                    for tot_grad, grad in zip((*(d[cout] for d in dw), dx_ws[(N, slice(None), *i)]), grad):
                        tot_grad += grad

                return *dw, dx

            return forward, backward
        return closure

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

    def conv2d(self, w, padding=0, strides=1):
        return self.sliding_window(self.dot, w, f_subscripts='...ijk,...ijk->...', \
        b_subscripts=('i...j,h...j->i...h', 'i...j,h...j->h...i'), kernel_size=w.shape[-2:], stride=1)

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