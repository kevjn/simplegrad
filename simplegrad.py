import functools, types, numpy as np
import typing
import itertools as it
import operator
from abc import abstractmethod
import pyopencl as cl
import pyopencl.array

class Device(object):
    class CPU:
        class Array(np.ndarray):
            def __new__(cls, input_array, symbolic=None):
                obj = np.array(input_array, dtype=np.float32).view(cls)
                obj.symbolic = str(symbolic or input_array)
                return obj

            def __array_finalize__(self, obj) -> None:
                if obj is None: return
                # This attribute should be maintained!
                self.symbolic = getattr(obj, 'symbolic', None)

            def __array_ufunc__(self, ufunc, method, *inputs, dtype=None, **kwargs):
                args = tuple(self._symbolic_args(inputs, kwargs))
                args += ("dtype=np.float32",) # require float32
                symbolic = f"np.{ufunc.__name__}({', '.join(args)})"

                items = (i.view(np.ndarray) if isinstance(i, Device.CPU.Array) else i for i in inputs)
                output = Device.CPU.Array(getattr(ufunc, method)(*items, **kwargs, dtype=dtype), symbolic)
                return output

            def __array_function__(self, func, types, inputs, kwargs):
                items = (i.view(np.ndarray) if isinstance(i, Device.CPU.Array) else i for i in inputs)
                symbolic = f"np.{func.__name__}({', '.join(self._symbolic_args(inputs, kwargs))})"
                out = Device.CPU.Array(func(*items, **kwargs), symbolic)

                return out

            def _symbolic_args(self, inputs, kwargs): 
                return it.chain(
                    (x.symbolic if isinstance(x, type(self)) else repr(x) for x in inputs),
                    (f'{x}={repr(y)}' for x,y in kwargs.items()),
                )

        def __getattr__(self, attr): return getattr(np, attr)

        def einsum(self, *operands, subscripts):
            return np.einsum(subscripts, *operands)

        def broadcast_to(self, x, y):
            return np.broadcast_to(x, y.shape)

        def relu(self, x): return np.maximum(x, 0)

        def as_strided(self, *args, **kwargs):
            return np.lib.stride_tricks.as_strided(*args, **kwargs)

        def to_device(self, x, name=None): return Device.CPU.Array(x, name)

        def to_cpu(self, x): return x.view(np.ndarray)

        # naming conventions
        pow = np.power
        mul = np.multiply

        @classmethod
        def arange(cls, n):
            arr = np.arange(n)
            return cls.Array(arr, np.array2string(arr.astype(np.float32), separator=', '))

    class GPU:
        class Array:
            def __init__(self, shape):
                self.dtype = np.dtype(np.float32)
                self.size = int(np.prod(shape))
                self.shape = shape
                self.strides = (self.dtype.itemsize, *np.multiply.accumulate(shape[:0:-1]) * self.dtype.itemsize)[::-1]
                self.ndim = len(shape)
                self.nbytes = self.dtype.itemsize * self.size
                self.symbolic = 'temp'

                assert self.strides == np.empty(tuple(shape)).astype(np.float32).strides

            def __repr__(self):
                return repr(self.get())

            for dunder in ('add', 'pow', 'mul'): # temp fix
                locals()[f'__{dunder}__'] = lambda self, x, __f=dunder: getattr(Tensor.device, __f)(self, x)
            
            def __sub__(self, other): return Tensor.device.add(self, Tensor.device.mul(other, Tensor.device.to_device(-1)))
            def __truediv__(self, other): return Tensor.device.mul(self, Tensor.device.pow(other, Tensor.device.to_device(-1)))

            def get(self):
                arr = np.empty(self.shape, np.float32)
                cl.enqueue_copy(Device.GPU.queue, arr, self.data)
                return arr

            def reshape(self, shape):
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

            def transpose(self, order):
                self.shape = tuple(np.take(self.shape, order))
                self.strides = tuple(np.take(self.strides, order))
                return self

            @classmethod
            def from_numpy(cls, arr):
                arr = arr.astype(np.float32) # atleast1d
                obj = cls(arr.shape)
                obj.data = cl.Buffer(Device.GPU.ctx, cl.mem_flags.READ_WRITE |
                              cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
                return obj

            @classmethod
            def empty(cls, shape):
                obj = cls(shape)
                obj.data = cl.Buffer(Device.GPU.ctx, cl.mem_flags.READ_WRITE, obj.nbytes)
                return obj

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
                wrapped_gpu_op = functools.partial(parser, functools.partial(kernel, self.queue))
                setattr(self, name, wrapped_gpu_op)

        def to_device(self, x, name=None):
            if isinstance(x, Device.GPU.Array):
                return x.copy()
            return self.Array.from_numpy(np.atleast_1d(x))

        def to_cpu(self, x):
            return x.get()

        def reshape(self, x, shape):
            return x.reshape(shape)

        @classmethod
        def arange(self, n):
            return Device.GPU.Array.from_numpy(np.arange(n))

        class Parser(object):
            def elementwise(kernel, *args, **kwargs):
                # allocate output buffer on device
                res = Device.GPU.Array.empty(args[0].shape)
                kernel([args[0].size], None, *(a.data for a in (*args, res)))
                return res

            def broadcast(kernel, *args, **kwargs):
                assert all(arg.ndim > 0 for arg in args), "operands needs to be atleast 1d"
                res_shape = np.broadcast_shapes(*(x.shape for x in args))
                res = Device.GPU.Array.empty(res_shape)
                res_strides = cl.array.to_device(Device.GPU.queue, np.array(res.strides, dtype=np.int32) // 4)

                strides = ((cl.array.to_device(Device.GPU.queue, 
                            (np.equal(np.pad(x.shape, (res.ndim-x.ndim, 0)), res.shape) * 
                            np.pad(x.strides, (res.ndim-x.ndim, 0)) // 4).astype(np.int32)).data)
                            for x in args)

                args = it.chain(*zip((a.data for a in args), strides), (res.data, np.int32(res.ndim), res_strides.data))

                kernel(res.shape, None, *args)
                return res

            def einsum(kernel, x, y, subscripts):
                # combines broadcasting and reduction parsing
                assert subscripts
                x_subs, y_subs, out_subs = subscripts.replace('->',',').split(',')

                # parse ellipsis if needed
                if '...' in subscripts:
                    x_subs, y_subs = (subs.replace('...', str().join(map(chr, \
                        range(97, 97 + nd-sum(map(len, subs.split('...'))))))) \
                        for nd, subs in [(x.ndim, x_subs), (y.ndim, y_subs)])

                    # TODO: this will not work in all cases
                    out_subs = max(x_subs, y_subs, key=len)

                reduced_subscripts = (set(x_subs) | set(y_subs)) - set(out_subs)

                xstrides, ystrides = (np.floor_divide(s, 4, dtype=np.int32) for s in (x.strides, y.strides))

                # deduce output shape
                res_shape = tuple((s in x_subs and x.shape[x_subs.index(s)]) \
                          or (s in y_subs and y.shape[y_subs.index(s)]) for s in out_subs)

                # insert any reduced subscripts in center of out_subs
                L,R = ["".join(chr(c) for c in s) for s in np.array_split([ord(s) for s in out_subs], 2)]
                out_subs = "".join((L, *reduced_subscripts, R))

                # argsort operands relative to out_subs
                xorder, yorder = (sorted(range(len(subs)), key=lambda k: out_subs.index(subs[k])) for subs in (x_subs, y_subs))

                if not reduced_subscripts:
                    x = x.transpose(xorder)
                    y = y.transpose(yorder)

                    # standard multiplication
                    return Tensor.device.mul(x, y)

                assert len(reduced_subscripts) == 1, "reduction over multiple axis not implemented yet"
                reduced_subscript = reduced_subscripts.pop()

                # reduced dimension in operands
                reduced_axis_x = x_subs.index(reduced_subscript)
                reduced_axis_y = y_subs.index(reduced_subscript)

                # corresponding stride
                reduced_axis_stride_x = xstrides[reduced_axis_x]
                reduced_axis_stride_y = ystrides[reduced_axis_y]

                assert x.shape[reduced_axis_x] == y.shape[reduced_axis_y]
                reduced_axis_size = x.shape[reduced_axis_x]

                # set stride of reduced dimension to 0 and sort
                np.put(xstrides, reduced_axis_x, 0)
                xstrides = xstrides[xorder]
                np.put(ystrides, reduced_axis_y, 0)
                ystrides = ystrides[yorder]

                # extend strides with broadcasted dimensions if needed
                broadcasted_subs_y = len(y_subs) < len(x_subs) and set(x_subs) - set(y_subs)
                if broadcasted_subs_y:
                    axis = np.take(xorder, [x_subs.index(s) for s in broadcasted_subs_y])
                    # trim if needed
                    axis = axis[(y.ndim + axis.size) - x.ndim:]
                    ystrides = np.insert(ystrides, axis, 0)

                broadcasted_subs_x = len(x_subs) < len(y_subs) and set(y_subs) - set(x_subs)
                if broadcasted_subs_x:
                    axis = np.take(yorder, [y_subs.index(s) for s in broadcasted_subs_x])
                    xstrides = np.insert(xstrides, axis, 0)

                res = Device.GPU.Array.empty(res_shape)

                res_strides = not res.ndim < max(x.ndim, y.ndim) and res.strides or \
                    tuple(np.insert(res.strides, out_subs.index(reduced_subscript), 0))
                res_strides = np.array(res_strides, dtype=np.int32) // 4

                # convert to opencl
                reduced_axis_size = np.int32(reduced_axis_size)
                x_strides = cl.array.to_device(Device.GPU.queue, xstrides)
                y_strides = cl.array.to_device(Device.GPU.queue, ystrides)

                reduced_axis_stride_x = np.int32(reduced_axis_stride_x)
                reduced_axis_stride_y = np.int32(reduced_axis_stride_y)

                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)

                max_shape = tuple(map(lambda x: x[0] or max(x[1], x[2]), 
                                it.zip_longest(res.shape, x.shape, y.shape, fillvalue=0)))

                # call kernel
                kernel(max_shape, None, x.data, y.data, x_strides.data, y_strides.data, \
                    reduced_axis_stride_x, reduced_axis_stride_y, reduced_axis_size, res.data, res_strides.data)

                return res

            def axis(x, ax):
                return (*range(x.ndim),) if ax is None else ax if type(ax) is tuple else ax if ax >= 0 else x.ndim + ax

            def reduce(kernel, x, axis=None, keepdims=False, **kwargs):
                axis = Device.GPU.Parser.axis(x, axis)

                if type(axis) is tuple:
                    for ax in sorted(axis, reverse=True):
                        x = Device.GPU.Parser.reduce(kernel, x, axis=ax)
                    return x

                strides = np.array(x.strides, dtype=np.int32) // (x.nbytes // x.size)
                strides = cl.array.to_device(Device.GPU.queue, strides)

                anchored_axes = np.delete(range(x.ndim), axis).astype(np.int32)
                if not anchored_axes.size:
                    anchored_axes = np.array([axis], dtype=np.int32)
                    global_work_size = np.array([1], dtype=np.int32)
                else:
                    global_work_size = np.array(x.shape, dtype=np.int32)[anchored_axes]

                anchored_axes = cl.array.to_device(Device.GPU.queue, anchored_axes)

                res = Device.GPU.Array.empty((*global_work_size, *[1] * keepdims * (x.ndim - global_work_size.size)))
                res_strides = np.array(res.strides, dtype=np.int32) // (res.nbytes // res.size)
                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)
                
                args = x.data, strides.data, anchored_axes.data, \
                       np.int32(axis), np.int32(x.shape[axis]), res.data, res_strides.data

                kernel(global_work_size, None, *args)
                return res

# pretty print arrays
# np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

class Tensor(object):
    device = Device.CPU()

    def __init__(self, data, name=None):
        self.data = Tensor.device.to_device(data, name)
        self.grad = 0

        self.backward_fxns = [] 
        self.arguments = []

    def __repr__(self):
        return f"Tensor({self.data!s})"

    def cpu(self):
        return Tensor.device.to_cpu(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return self.data[idx]

    def _backward(self, dv):
        self.grad = dv
        while self.backward_fxns:
            backward = self.backward_fxns.pop()
            *args, kwargs = self.grad, *self.arguments.pop()
            self.grad = backward(*args, **kwargs)

        assert not type(self.grad) is tuple

    def backward(self):
        # implicit gradient creation
        self._backward(Tensor.device.to_device(np.ones(self.shape, dtype=np.float32)))

    def __getattr__(self, attr):
        forward = getattr(Tensor.device, attr)
        backward = getattr(Tensor, f"{attr}_backward")

        def wrapper(*operands, **kwargs):
            args = tuple(x.data for x in (self, *operands))
            self.data = forward(*args, **kwargs)

            # save intermediate variables and (output) for backward pass
            self.arguments.append((*operands, *args, self.data, kwargs))
            self.backward_fxns.append(backward)

            return self
        return wrapper

    def unbroadcast(backward):
        def reduce(grad, shape):
            _shape = (-1,) * abs(len(shape)-len(grad.shape)) + shape
            idx = np.not_equal(grad.shape, _shape)
            axes = *np.arange(grad.ndim)[idx],
            if not axes:
                return grad
            return Tensor.device.reshape(Tensor.device.sum(grad, axis=axes), shape)
        @functools.wraps(backward)
        def wrapper(*args, shapes, **kwargs):
            return tuple(reduce(*args) for args in zip(backward(*args, **kwargs), shapes))
        return wrapper

    def propagate(backward):
        @functools.wraps(backward)
        def wrapper(dv, operand, x, y, out, **kwargs):
            dy, dx = backward(dv, x, y, out, shapes=(y.shape, x.shape), **kwargs)
            operand._backward(dy)
            return dx
        return wrapper

    # ========== unary ops ==========

    def relu_backward(dv, x, out): 
        return Tensor.device.mul(dv, Tensor.device.greater_equal(x, Tensor.device.to_device(0)))

    def exp_backward(dv, x, out):
        return Tensor.device.mul(dv, out)

    def log_backward(dv, x, out):
        return Tensor.device.mul(dv, Tensor.device.pow(x, Tensor.device.to_device(-1)))
    
    # ========== reduce ops ==========

    def sum_backward(dv, x, out, axis=None, keepdims=False):
        if x.ndim > dv.ndim:
            dv = Tensor.device.reshape(dv, (*dv.shape, 1))
        return Tensor.device.broadcast_to(dv, x)
    
    def max_backward(dv, x, out, axis=None, keepdims=False):
        if keepdims:
            dv = dv.squeeze() # remove empty dims
        r = Tensor.device.reshape(x, (*dv.shape, -1)) # flatten reduced axes
        max_idx = Tensor.device.argmax(r, axis=-1)

        # add one empty dimension for broadcasting
        max_idx = Tensor.device.reshape(max_idx, (*max_idx.shape, 1))
        dv = Tensor.device.reshape(dv, (*dv.shape, 1))

        mask = Tensor.device.equal(max_idx, Tensor.device.arange(r.shape[-1]))
        r = Tensor.device.mul(mask, dv)
        return Tensor.device.reshape(r, x.shape)

    # ========== binary ops ==========

    def pow_backward(dv, operand, x, y, out):
        return Tensor.device.mul(dv, Tensor.device.mul(y, 
               Tensor.device.pow(x, Tensor(y).sub(Tensor(1.0)).data)))

    @propagate
    @unbroadcast
    def add_backward(dv, x, y, out):
        return dv, dv

    @propagate
    @unbroadcast
    def mul_backward(dv, x, y, out):
        return Tensor.device.mul(dv, x), Tensor.device.mul(dv, y)

    # ========== processing ops ==========
    
    @propagate
    @unbroadcast
    def einsum_backward(dv, x, y, out, subscripts='i...j,j...k->i...k'):
        input_subs, output_subs = subscripts.split('->')
        x_subs, y_subs = input_subs.split(',')
        reduced_subscripts_x = set(x_subs) - set(output_subs + y_subs)
        x_subs_non_reduced = "".join(filter(lambda x: x not in reduced_subscripts_x, x_subs))
        dx = Tensor.device.einsum(dv, y, subscripts=f"{output_subs},{y_subs}->{x_subs_non_reduced}")

        reduced_subscripts_y = set(y_subs) - set(output_subs + x_subs)
        y_subs_non_reduced = "".join(filter(lambda y: y not in reduced_subscripts_y, y_subs))
        dy = Tensor.device.einsum(dv, x, subscripts=f"{output_subs},{x_subs}->{y_subs_non_reduced}")

        if reduced_subscripts_x:
            raise NotImplementedError # Tiling not implemented

        if reduced_subscripts_y:
            raise NotImplementedError # Tiling not implemented

        return dy, dx

    def as_strided_backward(dv, x, out, **kwargs):
        assert dv.shape == out.shape
        accum = np.zeros(x.shape)
        kdims = out.ndim - x.ndim
        kernel_size = out.shape[-kdims:]

        slices = (slice(None),) * 2 # the first two dims are fixed
        for idx in np.ndindex(out.shape[2:2+kdims]):
            ws_slices = tuple(slice(i, i+sz) for i,sz in zip(idx, kernel_size))
            accum[slices + ws_slices] += dv[slices + idx]

        return accum

    # ========== composite ops ==========

    def window_view(self, kernel_size=(2,2), stride=1):
        N, cin, *in_shape, kdims = *self.shape, len(kernel_size)

        # get window shape and strides
        truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
        out_shape = N, cin, *truncated_out_shape, *kernel_size
        out_strides = *self.data.strides[:2], *(xs*stride for xs in self.data.strides[-kdims:]), *self.data.strides[-kdims:]

        # return window view
        return self.as_strided(shape=out_shape, strides=out_strides)

    def einsum(self, subscripts, *operands):
        return self.__getattr__('einsum')(*operands, subscripts=subscripts)

    def div(self, x):
        assert isinstance(x, type(self))
        return self.mul(x.pow(Tensor(-1)))

    def sub(self, x):
        return self.add(x.mul(Tensor(-1)))

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

    def mean(self, axis=-1):
        a = Tensor(np.prod(np.take(self.shape, axis)))
        return self.sum(axis=axis, keepdims=True).div(a)

    def layer_norm(self, axes, weight, bias, eps=1e-5):
        mean = self.fork().mean(axis=axes)
        self.sub(mean)
        sd = self.fork().pow(Tensor(2)).mean(axis=axes)
        denom = sd.add(Tensor(eps)).pow(Tensor(0.5))
        return self.div(denom).mul(weight).add(bias)

    def conv2d(self, w, padding=0, stride=1):
        assert len(self.shape) == len(w.shape) == 4
        return self.window_view(kernel_size=w.shape[-2:], stride=stride).einsum('abcdef,gbef->agcd', w)

    def maxpool2d(self, kernel_size = (3,3), padding=0, stride=1):
        return self.window_view(kernel_size=kernel_size, stride=stride).max(axis=(-1, -2))

    def sigmoid(self):
        return self.exp().pow(Tensor(-1)).add(Tensor(1)).pow(Tensor(-1))

    def tanh(self):
        e2, e3, e4 = self.fork().mul(Tensor(-1)).exp(), self.fork().exp(), \
                     self.fork().mul(Tensor(-1)).exp()

        return self.exp().sub(e2).div(e3.add(e4))

    # ========== control flow ops ==========

    def fork(self):

        dv_fork = None

        def fork_backward(dv):
            nonlocal dv_fork
            dv_fork = dv
            return dv

        fork = Tensor(self.data, name=self.data.symbolic)
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
        self.args = tuple(map(Tensor.device.to_device, args))
        self.t = Tensor.device.to_device(0)

    def step(self):
        self.t = Tensor.device.add(self.t, Tensor.device.to_device(1))
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

        self.m = [Tensor.device.to_device(np.zeros(p.shape)) for p in params]
        self.v =  [Tensor.device.to_device(np.zeros(p.shape)) for p in params]

    def _step(self, t, lr, eps, b1, b2):
        td = Tensor.device.to_device # TODO: make this implicit
        # bias correction
        lr = lr * ((td(1) - b2**t)**td(0.5)) / (td(1.0) - b1**t)
        
        for i,t in enumerate(self.params):
            self.m[i] = b1 * self.m[i] + (td(1.0) - b1) * t.grad
            self.v[i] = b2 * self.v[i] + (td(1.0) - b2) * t.grad * t.grad
            t.data = t.data - lr * self.m[i] / (self.v[i] ** td(0.5) + eps)