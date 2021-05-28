import functools, types, numpy as np
import typing
import itertools as it
import operator
from abc import abstractmethod
import pyopencl as cl
import pyopencl.array

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


class Device(object):
    class CPU:
        class SymbolicArray(np.ndarray):
            def __new__(cls, arr, symbolic=None, **kwargs):
                arr = np.array(arr, copy=False, **kwargs)
                obj = arr.view(cls)
                obj.symbolic = str(symbolic or np.array2string(arr, separator=',', threshold=np.inf, floatmode='unique'))
                return obj

            def __array_finalize__(self, obj) -> None:
                if obj is None: return
                # This attribute should be maintained!
                self.symbolic = getattr(obj, 'symbolic', None)

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                assert method == '__call__'
                return np.core.overrides.array_function_dispatch \
                    (lambda *args, **kwargs: args, verify=False, module='numpy') (ufunc) \
                    (*inputs, **kwargs)

            def __array_function__(self, func, types, inputs, kwargs):
                assert func.__module__
                return Device.CPU.SymbolicArray (
                    func (
                        *(x.view(np.ndarray) if isinstance(x, Device.CPU.SymbolicArray) else x for x in inputs),
                        **kwargs
                    ),
                    f"{func.__module__}.{func.__name__}({', '.join(self._symbolic_args(inputs, kwargs))})"
                )

            def _symbolic_args(self, inputs, kwargs): 
                return it.chain (
                    (x.symbolic if isinstance(x, type(self)) else repr(x) for x in inputs),
                    (f'{x}={repr(y)}' for x,y in kwargs.items()),
                )

    class GPU:
        class Array(np.lib.mixins.NDArrayOperatorsMixin):

            def __init__(self, shape, dtype=np.float32, data=None):
                self.shape = shape
                self.dtype = np.dtype(dtype)
                self.size = int(np.prod(shape))
                self.strides = tuple(np.multiply.accumulate([1, *shape[:0:-1]]) * self.dtype.itemsize)[::-1]
                self.ndim = len(shape)
                self.nbytes = self.dtype.itemsize * self.size
                self.data = data
                self.base = None

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                assert method == '__call__'
                return self.__array_function__(ufunc, None, inputs, kwargs)

            def __array_function__(self, func, types, inputs, kwargs):
                func = getattr(Device.GPU, func.__name__)
                return func(*inputs, **kwargs)

            def __repr__(self):
                return f"GPUArray({np.array2string(self.get(), 88, 4, True, ', ', 'GPUArray(', suffix=')')})"

            def get(self):
                res = np.empty(self.data.size // 4, self.dtype)
                cl.enqueue_copy(Device.GPU.queue, res, self.data)
                if not self.shape:
                    return res
                return np.lib.stride_tricks.as_strided(res, self.shape, self.strides)

            def reshape(self, *shape):
                if -1 in shape:
                    shape = tuple(x if x > 0 else 
                            int(abs(np.prod(self.shape) / np.prod(shape)))
                            for x in shape)
                result = Device.GPU.Array(shape)
                result.data = self.data
                return result

            def squeeze(self):
                shape = tuple(np.compress(np.array(self.shape) > 1, self.shape))
                result = Device.GPU.Array(shape)
                result.data = self.data
                return result

            def transpose(self, *order):
                shape = tuple(np.take(self.shape, order))
                result = self.__class__(shape)
                result.strides = tuple(np.take(self.strides, order))
                result.data = self.data
                return result

            def copy(self):
                result = Device.GPU.Array(self.shape)
                data = cl.Buffer(Device.GPU.ctx, cl.mem_flags.READ_WRITE, self.nbytes)
                cl.enqueue_copy(Device.GPU.queue, data, self.data, byte_count=self.nbytes)
                result.data = data
                return result

        def __init__(self):
            # initialize opencl
            Device.GPU.ctx = cl.create_some_context()
            Device.GPU.queue = cl.CommandQueue(self.ctx)

            prg = cl.Program(Device.GPU.ctx, open('./accelerators/gpu_ops.cl').read()).build()
            for kernel in prg.all_kernels():
                tokens = kernel.function_name.split("__")
                assert len(tokens) == 2
                name, parser = tokens
                parser = getattr(self.Parser, parser)
                wrapped_gpu_op = self.Parser.wrapper(parser, functools.partial(kernel, self.queue))
                setattr(Device.GPU, name, wrapped_gpu_op)

        def to_cpu(x):
            return x.get()

        def reshape(x, shape):
            if x.base:
                return Device.GPU.broadcast_to(x, shape)
            return x.reshape(*shape)

        def broadcast_to(x, shape):
            if x.base:
                x = x.base

            # set strides to 0 for all singleton dimensions
            strides = np.where(np.equal(x.shape, 1), 0, x.strides)
            # add empty trailing strides if needed
            strides = np.append(strides, np.array([0]*abs(x.ndim - len(shape)), int))
            
            arr = Device.GPU.Array(shape)
            arr.data = x.data
            arr.strides = tuple(strides)
            arr.base = x
            return arr

        def array(arr, dtype=np.float32, ndmin=1, **kwargs):
            arr = np.array(arr, copy=False, dtype=dtype, ndmin=ndmin, **kwargs)
            if arr.size:
                data = cl.Buffer(Device.GPU.ctx, cl.mem_flags.READ_WRITE |
                                    cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
            else:
                data = None
            return Device.GPU.Array(arr.shape, dtype, data)

        def empty(shape, dtype=np.float32):
            arr = Device.GPU.Array(shape, dtype)
            arr.data = cl.Buffer(Device.GPU.ctx, cl.mem_flags.READ_WRITE, arr.nbytes)
            return arr

        def arange(n):
            return Device.GPU.array(np.arange(n), dtype=np.int32)

        class Parser(object):
            def wrapper(parser, kernel):
                def _wrapper(*args, **kwargs):
                    args = tuple(x if isinstance(x, (str, Device.GPU.Array)) 
                                else Device.GPU.array(x) for x in args)
                    return parser(kernel, *args, **kwargs)
                return _wrapper
            
            def elementwise(kernel, *args, **kwargs):
                # allocate output buffer on device
                res = Device.GPU.empty(args[0].shape)
                kernel([args[0].size], None, *(a.data for a in (*args, res)))
                return res

            def broadcast(kernel, x, y):
                assert x.ndim > 0 and y.ndim > 0, "operands needs to be atleast 1d"

                res_shape = np.broadcast_shapes(x.shape, y.shape)
                xstrides = np.arange(np.prod(x.shape), dtype=np.int32).reshape(x.shape)
                ystrides = np.arange(np.prod(y.shape), dtype=np.int32).reshape(y.shape)

                xstrides.strides = x.strides
                ystrides.strides = y.strides

                xstrides = np.broadcast_to(xstrides, res_shape).flatten()
                ystrides = np.broadcast_to(ystrides, res_shape).flatten()

                res = Device.GPU.empty(res_shape)
                res_strides = np.arange(np.prod(res_shape))
                
                # convert to opencl
                strides = [cl.array.to_device(Device.GPU.queue, x.astype(np.int32)) for x in 
                                             (xstrides, ystrides, res_strides)]

                args = (x, y, res)
                args = tuple(it.chain(*zip((a for a in args), strides)))

                kernel([np.prod(res_shape)], None, *(arg.data for arg in args))

                return res

            def einsum(kernel, subscripts, x, y):
                # combines broadcasting and reduction parsing
                x_subs, y_subs, out_subs = subscripts.replace('->',',').split(',')

                # parse ellipsis if needed
                if '...' in subscripts:
                    x_subs, y_subs = (subs.replace('...', str().join(map(chr, \
                        range(97, 97 + nd-sum(map(len, subs.split('...'))))))) \
                        for nd, subs in [(x.ndim, x_subs), (y.ndim, y_subs)])

                    # TODO: this will not work in all cases
                    out_subs = max(x_subs, y_subs, key=len)

                # deduce output shape
                res_shape = tuple([y.shape[y_subs.find(s)], x.shape[x_subs.find(s)]][s in x_subs] for s in out_subs)

                reduced_subscripts = list((set(x_subs) | set(y_subs)) - set(out_subs))
                if not reduced_subscripts:
                    # transpose operands relative to out_subs
                    x = x.transpose(*[out_subs.index(x) for x in x_subs])
                    y = y.transpose(*[out_subs.index(x) for x in y_subs])

                    # standard multiplication
                    return Device.GPU.multiply(x, y)

                xstrides = np.arange(np.prod(x.shape), dtype=np.int32)
                stride = [int(s in x_subs and x.strides[x_subs.index(s)]) for s in out_subs]
                xstrides = np.lib.stride_tricks.as_strided(xstrides, res_shape, stride).copy()

                ystrides = np.arange(np.prod(y.shape), dtype=np.int32)
                stride = [int(s in y_subs and y.strides[y_subs.index(s)]) for s in out_subs]
                ystrides = np.lib.stride_tricks.as_strided(ystrides, res_shape, stride).copy()

                # reduced dimension in operands
                reduced_shape = tuple([y.shape[y_subs.find(s)], x.shape[x_subs.find(s)]][s in x_subs] for s in reduced_subscripts)

                reduced_axes_stride_x = [int(s in x_subs and x.strides[x_subs.index(s)]) for s in reduced_subscripts]
                stride = np.arange(np.prod(x.shape), dtype=np.int32)
                reduced_axes_stride_x = np.lib.stride_tricks.as_strided(stride, reduced_shape, reduced_axes_stride_x).copy()

                reduced_axes_stride_y = [int(s in y_subs and y.strides[y_subs.index(s)]) for s in reduced_subscripts]
                stride = np.arange(np.prod(y.shape), dtype=np.int32)
                reduced_axes_stride_y = np.lib.stride_tricks.as_strided(stride, reduced_shape, reduced_axes_stride_y).copy()

                reduced_axis_size = np.prod(reduced_shape)

                res = Device.GPU.empty(res_shape)
                res_strides = np.arange(np.prod(res_shape), dtype=np.int32)

                # convert to opencl
                reduced_axis_size = np.int32(reduced_axis_size)
                x_strides = cl.array.to_device(Device.GPU.queue, xstrides)
                y_strides = cl.array.to_device(Device.GPU.queue, ystrides)

                reduced_axis_stride_x = cl.array.to_device(Device.GPU.queue, reduced_axes_stride_x)
                reduced_axis_stride_y = cl.array.to_device(Device.GPU.queue, reduced_axes_stride_y)

                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)

                # call kernel
                kernel([np.prod(res_shape)], None, x.data, y.data, x_strides.data, y_strides.data, \
                    reduced_axis_stride_x.data, reduced_axis_stride_y.data, reduced_axis_size, res.data, res_strides.data)

                return res

            def reduce(kernel, x, axis=None, keepdims=False):
                axis = tuple(np.arange(x.ndim)[tuple([axis])].flatten())

                meta = np.stack([x.shape, x.strides]) 
                reduced_shape, reduced_strides = meta[:,axis]
                result_shape, xstrides = np.delete(meta, axis, axis=1)

                strides = np.arange(np.prod(x.shape), dtype=np.int32)
                reduced_axes_stride = np.lib.stride_tricks.as_strided(strides, reduced_shape, reduced_strides).copy()
                xstrides = np.lib.stride_tricks.as_strided(strides, result_shape, xstrides).copy()

                if keepdims:
                    np.put(meta[0], axis, 1)
                    result_shape = meta[0]
                if not result_shape.size:
                    result_shape = (1,)
                result = Device.GPU.empty(tuple(result_shape))
                result_strides = np.arange(np.prod(result_shape), dtype=np.int32)

                # convert to opencl
                reduced_axes_stride = cl.array.to_device(Device.GPU.queue, reduced_axes_stride)
                xstrides = cl.array.to_device(Device.GPU.queue, xstrides)
                reduced_axis_size = np.int32(np.prod(reduced_shape))
                result_strides = cl.array.to_device(Device.GPU.queue, result_strides)

                args = x.data, xstrides.data, reduced_axes_stride.data, reduced_axis_size, \
                        result.data, result_strides.data

                kernel([np.prod(result_shape)], None, *args)

                return result


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    device = Numpy

    def __init__(self, data):
        assert isinstance(data, type(Tensor.device.array([])))
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
        assert self.data.size == 1, 'output must be scalar for implicit grad creation'
        # implicit gradient creation
        self._backward(Tensor.device.array(1.0, ndmin=self.data.ndim))

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

        mask = Tensor.device.equal(max_idx, Tensor.device.arange(r.shape[-1], like=Tensor.device.array([])))
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
        indices = Tensor.device.arange(np.prod(x.shape), like=Tensor.device.array([]))

        kwargs['strides'] = tuple(indices.strides * (kwargs['strides'] // np.min(kwargs['strides'])))
        indices = Tensor.device.as_strided(indices, **kwargs)

        # flatten operands
        indices, dv = Tensor.device.reshape(indices, -1), Tensor.device.reshape(dv, -1)

        return Tensor.device.reshape(Tensor.device.bincount(indices, dv), x.shape)

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
        assert isinstance(x, type(self))
        return self.mul(x.pow(-1.0))

    def sub(self, x):
        return self.add(x.mul(-1.0))

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
        div = Tensor.device.array(1/np.prod(np.take(self.shape, axis)))
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

        self.m = [Tensor.device.array(np.zeros(p.shape)) for p in params]
        self.v = [Tensor.device.array(np.zeros(p.shape)) for p in params]

    def _step(self, t, lr, eps, b1, b2):
        # bias correction
        lr = lr * ((1 - b2**t)**0.5) / (1.0 - b1**t)
        
        for i,t in enumerate(self.params):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * t.grad
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * t.grad * t.grad
            t.data = t.data - lr * self.m[i] / (self.v[i] ** 0.5 + eps)