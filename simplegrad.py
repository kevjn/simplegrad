import numpy as np
import functools

class NumpyType(type):
    def __getattr__(cls, attr):
        return getattr(np, attr)

class Numpy(metaclass=NumpyType):
    # np.lib.stride_tricks.as_strided does not use a dispatcher by default
    def as_strided(x, **kwargs):
        return np.core.overrides.array_function_dispatch \
                (lambda *args, **kwargs: args, verify=False) \
                (np.lib.stride_tricks.as_strided) (x, **kwargs)

    # naming conventions
    pow = np.power
    mul = np.multiply

    def to_cpu(x): return x.view(np.ndarray)

class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    device = Numpy

    def __init__(self, data):
        self.data = data
        self.grad = 0

        self.backward_fxns = [] 
        self.arguments = []

    def __repr__(self):
        return f"Tensor({np.array2string(self.cpu(), 88, 4, True, ', ', 'Tensor(', suffix=')')})"

    def cpu(self):
        return Tensor.device.to_cpu(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        def backward(dv, shape, idx):
            size = np.prod(shape)
            indices = Tensor.device.arange(size, like=Tensor.device.array([]))
            indices = Tensor.device.reshape(indices, shape)[idx]

            # flatten operands
            indices, dv = Tensor.device.reshape(indices, -1), Tensor.device.reshape(dv, -1)
            grad = Tensor.device.reshape(Tensor.device.bincount(indices, dv, minlength=size), shape)
            self._backward(grad)

        out = Tensor(self.data[idx])
        out.arguments.append((self.shape, idx, dict()))
        out.backward_fxns.append(backward)
        return out

    def _backward(self, dv):
        self.grad = dv
        while self.backward_fxns:
            backward = self.backward_fxns.pop()
            *args, kwargs = self.grad, *self.arguments.pop()
            self.grad = backward(*args, **kwargs)

        assert not type(self.grad) is tuple

    def backward(self):
        assert self.data.size == 1, 'output must be scalar for implicit gradient creation'
        # implicit gradient creation
        self._backward(Tensor.device.array(1.0, ndmin=self.data.ndim, like=type(self.data)([])))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == '__call__'
        return self.__array_function__(ufunc, None, inputs, kwargs)

    def __array_function__(self, func, types, inputs, kwargs):
        backward = getattr(Tensor, f"{func.__name__}_backward")

        args = tuple(x.data if isinstance(x, Tensor) else x for x in inputs)
        meta = tuple(x for x in inputs if isinstance(x, Tensor))

        self.data = out = func(*args, **kwargs)

        # save intermediate variables and (output) for backward pass
        self.arguments.append((meta, *args, out, kwargs))
        self.backward_fxns.append(backward)

        return self

    def __getattr__(self, attr):
        return functools.partial(getattr(Tensor.device, attr), self)

    def unbroadcast(backward):
        # summation is the dual of broadcasting
        def reduce(grad, inp):
            shape = inp.shape # np.asarray(inp).shape
            mask = np.insert(shape, 0, [-1]*abs(grad.ndim - len(shape))) != grad.shape
            if axis := tuple( np.r_[:grad.ndim][mask] ):
                return Tensor.device.reshape(Tensor.device.sum(grad, axis=axis), shape)
            return grad
        @functools.wraps(backward)
        def wrapper(dv, x, y, out, **kwargs):
            yield from (reduce(*args) for args in zip(backward(dv, x, y, out, **kwargs), (x, y)))
        return wrapper

    def propagate(backward):
        @functools.wraps(backward)
        def wrapper(dv, operands, *args, **kwargs):
            backward_ = backward(dv, *args, **kwargs)
            dx = next(backward_)
            for operand in operands[1:]:
                operand._backward(next(backward_))
            return dx
        return wrapper

    # ========== unary ops ==========

    def maximum_backward(dv, meta, x, y, out): 
        return dv * (x >= y)

    def exp_backward(dv, meta, x, out):
        return dv * out

    def log_backward(dv, meta, x, out):
        return dv * x ** -1
    
    # ========== reduce ops ==========

    def sum_backward(dv, meta, x, out, axis=None, keepdims=False):
        if x.ndim > dv.ndim:
            dv = Tensor.device.reshape(dv, (*dv.shape, 1))
        return Tensor.device.broadcast_to(dv, x.shape)
    
    def amax_backward(dv, meta, x, out, axis=None, keepdims=False):
        if keepdims:
            dv = dv.squeeze() # remove empty dims
        r = Tensor.device.reshape(x, (*dv.shape, -1)) # flatten reduced axes
        max_idx = Tensor.device.argmax(r, axis=-1)

        # add one empty dimension for broadcasting
        max_idx = Tensor.device.reshape(max_idx, (*max_idx.shape, 1))
        dv = Tensor.device.reshape(dv, (*dv.shape, 1))

        mask = Tensor.device.equal(max_idx, Tensor.device.arange(r.shape[-1], like=type(dv)([])))
        return Tensor.device.reshape(mask * dv, x.shape)

    # ========== binary ops ==========

    def pow_backward(dv, operand, x, y, out):
        return dv * y * x ** (y-1)
    power_backward = pow_backward

    @propagate
    @unbroadcast
    def add_backward(dv, x, y, out):
        yield from (dv, dv)

    @propagate
    @unbroadcast
    def mul_backward(dv, x, y, out):
        yield dv * y
        yield dv * x
    multiply_backward = mul_backward

    # ========== processing ops ==========
    
    @propagate
    def einsum_backward(dv, subscripts, x, y, out):
        input_subs, output_subs = subscripts.split('->')
        x_subs, y_subs = input_subs.split(',')
        reduced_subscripts_x = set(x_subs) - set(output_subs + y_subs)
        x_subs_non_reduced = "".join(filter(lambda x: x not in reduced_subscripts_x, x_subs))
        yield Tensor.device.einsum(f"{output_subs},{y_subs}->{x_subs_non_reduced}", dv, y)

        reduced_subscripts_y = set(y_subs) - set(output_subs + x_subs)
        y_subs_non_reduced = "".join(filter(lambda y: y not in reduced_subscripts_y, y_subs))
        yield Tensor.device.einsum(f"{output_subs},{x_subs}->{y_subs_non_reduced}", dv, x)

    def as_strided_backward(dv, meta, x, out, **kwargs):
        assert dv.shape == out.shape
        indices = Tensor.device.arange(np.prod(x.shape), like=type(dv)([]))

        kwargs['strides'] = tuple(indices.strides * (kwargs['strides'] // np.min(kwargs['strides'])))
        indices = Tensor.device.as_strided(indices, **kwargs).copy()

        # flatten operands
        indices, dv = Tensor.device.reshape(indices, -1), Tensor.device.reshape(dv, -1)

        return Tensor.device.reshape(Tensor.device.bincount(indices, dv, minlength=x.size), x.shape)

    # ========== composite ops ==========

    def relu(self):
        return self.maximum(0)

    def window_view(self, kernel_size=(2,2), stride=1):
        N, cin, *in_shape, kdims = *self.shape, len(kernel_size)

        strides = self.data.strides
        # get window shape and strides
        truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
        out_shape = N, cin, *truncated_out_shape, *kernel_size
        out_strides = *strides[:2], *(xs*stride for xs in strides[-kdims:]), *strides[-kdims:]

        # return window view
        return self.as_strided(shape=out_shape, strides=out_strides)

    def einsum(self, subscripts, *operands):
        return Tensor.device.einsum(subscripts, self, *operands)

    def div(self, x):
        return self.mul(x ** -1.0)

    def sub(self, x):
        return self.add(x * -1.0)

    def softmax(self):
        _max = self.fork().max(axis=-1, keepdims=True)
        self.sub(_max).exp()
        s = self.fork().sum(axis=-1, keepdims=True)
        return self.div(s)

    def logsoftmax(self):
        _max = self.fork().max(axis=-1, keepdims=True)
        _sub = self.fork().sub(_max.fork()).exp().sum(axis=-1, keepdims=True).log()
        _max.add(_sub)
        return self.sub(_max)

    def mean(self, axis=None):
        if not axis:
            axis = tuple(np.r_[:self.data.ndim])
        div = Tensor.device.array(1/np.prod(np.take(self.shape, axis)), like=type(self.data)([]))
        return self.sum(axis=axis, keepdims=True).mul(div)

    def layer_norm(self, axes, weight, bias, eps=1e-5):
        mean = self.fork().mean(axis=axes)
        self.sub(mean)
        sd = self.fork().pow(2).mean(axis=axes)
        denom = sd.add(eps).pow(0.5)
        return self.div(denom).mul(weight).add(bias)

    def conv2d(self, w, padding=0, stride=1):
        assert len(self.shape) == len(w.shape) == 4
        return self.window_view(kernel_size=w.shape[-2:], stride=stride).einsum('abcdef,gbef->agcd', w)

    def maxpool2d(self, kernel_size = (3,3), padding=0, stride=1):
        return self.window_view(kernel_size=kernel_size, stride=stride).max(axis=(-1, -2))

    def sigmoid(self):
        return self.exp().pow(-1).add(1).pow(-1)

    def tanh(self):
        max_exp = np.log(np.finfo(np.float32).max) # 88.72284
        self.data = self.data.clip(-max_exp, max_exp)

        e1, e2, e3, e4 = self,              self.fork().mul(-1).exp(), \
                         self.fork().exp(), self.fork().mul(-1).exp()

        return e1.exp().sub(e2).div(e3.add(e4))

    # ========== control flow ops ==========

    def fork(self):

        dv_fork = None

        def fork_backward(dv):
            nonlocal dv_fork
            dv_fork = dv
            return dv

        fork = Tensor(self.data)
        fork.backward_fxns.append(fork_backward)
        fork.arguments.append((dict(),)) # empty args

        def parent_backward(dv):
            nonlocal dv_fork
            assert dv_fork is not None
            return dv + dv_fork # accumulate parent gradient

        self.backward_fxns.append(parent_backward)
        self.arguments.append((dict(),)) # empty args

        return fork

class Optimizer:
    # TODO: fix decay
    def __init__(self, params, *args, decay=0.):
        self.params = params
        self.args = args
        self.t = 0

    def step(self):
        self.t += 1
        self._step(self.t, *self.args)

    def _step(self):
        raise NotImplementedError

    def zero_grad(self):
        for t in self.params:
            t.grad = 0.0

class Adam(Optimizer):
    
    def __init__(self, params, learning_rate = 0.01, epsilon = 1e-7, 
                       beta1 = 0.9, beta2 = 0.999, **kwargs):

        super().__init__(params, *(learning_rate, epsilon, beta1, beta2), **kwargs)

        self.m = [Tensor.device.array(np.zeros(p.shape)) for p in params]
        self.v = [Tensor.device.array(np.zeros(p.shape)) for p in params]

    def _step(self, t, lr, eps, b1, b2):
        # bias correction
        lr = lr * ((1 - b2**t)**0.5) / (1.0 - b1**t)
        
        for i,t in enumerate(self.params):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * t.grad
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * t.grad * t.grad
            t.data = t.data - lr * self.m[i] / (self.v[i] ** 0.5 + eps)