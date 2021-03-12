import functools, types, numpy as np
import typing
import itertools as it
import operator
import pyopencl as cl
import pyopencl.array
from collections import defaultdict

class Device(object):

    def load_device(device):
        device()
        Device.load = device.load
        Device.load_directly = device.load_directly
        Device.to_device = device.to_device
        # argument parser
        Device.Parser = device.Parser

    def load(cpu_expr, *parsers, **kwargs):
        # use cpu forward/backward fxn as default
        return cpu_expr(**kwargs) 

    def load_directly(name, parser, backward=False, **kwargs):
        # get unwrapped function
        f = getattr(Tensor, name).__wrapped__
        return f(**kwargs)[backward]

    def to_device(x):
        return np.array(x)

    def operation_wrapper(op, parser, **kwargs):
        def wrapper(*args):
            return op(*parser(*args, **kwargs))
        return wrapper

    class CPU:
        class Parser(object):
            def default(*args, **kwargs):
                return args

        Parser.reduction = Parser.binary = Parser.default
    Parser = CPU.Parser

    class GPU:
        def __init__(self):
            # initialize opencl
            Device.GPU.ctx = cl.create_some_context()
            Device.GPU.queue = cl.CommandQueue(self.ctx)

            prg = cl.Program(Device.GPU.ctx, open('./accelerators/gpu_ops.cl').read()).build()
            ops = defaultdict(lambda: [None, None])
            for kernel in prg.all_kernels():
                tokens = kernel.function_name.split("_")
                assert len(tokens) == 2
                name, direction = tokens
                ops[name][direction != "forward"] = Device.GPU.kernel_wrapper(kernel)

            # freeze dictionary values
            Device.GPU.ops = defaultdict(lambda: (None, None), \
                {name : tuple(v) for name, v in ops.items()})

        def load(cpu_expr, *parsers, **kwargs):
            # fall back on cpu if no implementation exists, TODO: memoize this
            return *(Device.operation_wrapper(a or b, a and parser or Device.CPU.Parser.default, **kwargs) \
                for a,b,parser in zip(Device.GPU.ops[cpu_expr.__name__], cpu_expr(**kwargs), parsers)),

        def load_directly(name, parser, backward=False, **kwargs):
            return Device.operation_wrapper(Device.GPU.ops[name][backward], parser, **kwargs)

        def to_device(x: np.ndarray):
            return cl.array.to_device(Device.GPU.queue, x.astype(np.float32))

        class Parser(object):

            def default(*args, **kwargs):
                # allocate output buffer on device
                res = cl.array.empty_like(args[0])
                return [args[0].size], None, res, *(a.data for a in args), res.data

            def binary(x, y, **kwargs):
                shapes_padded = it.zip_longest(x.shape[::-1], y.shape[::-1], fillvalue=0)
                res = cl.array.empty(Device.GPU.queue, tuple(it.starmap(max, shapes_padded))[::-1], np.float32)
                res_strides = cl.array.to_device(Device.GPU.queue, np.array(res.strides, dtype=np.int32) // 4)

                xstrides = (np.equal(np.pad(x.shape, (len(res.shape)-len(x.shape),0)), res.shape) * \
                            np.pad(x.strides, (len(res.shape)-len(x.shape),0)) // 4).astype(np.int32)
                xstrides = cl.array.to_device(Device.GPU.queue, xstrides)

                ystrides = (np.equal(np.pad(y.shape, (len(res.shape)-len(y.shape),0)), res.shape) * \
                            np.pad(y.strides, (len(res.shape)-len(y.shape),0)) // 4).astype(np.int32)
                ystrides = cl.array.to_device(Device.GPU.queue, ystrides)

                return res.shape, None, res, x.data, xstrides.data, y.data, ystrides.data, \
                        res.data, np.int32(res.ndim), res_strides.data

            def reduction(*args, axis=None, **kwargs):
                assert len(args) == 1, "binary reduction not implemented yet"
                assert axis is not None
                if type(axis) is tuple:
                    assert len(axis) == 1, "reduction over multiple axes not implemented yet"
                    axis = axis[0]

                x = args[0].astype(np.float32)

                strides = np.array(x.strides, dtype=np.int32) // (x.nbytes // x.size)
                strides = cl.array.to_device(Device.GPU.queue, strides)

                anchored_axes = np.arange(x.ndim, dtype=np.int32)
                anchored_axes = np.delete(anchored_axes, axis)
                if not anchored_axes.size:
                    anchored_axes = np.array([axis], dtype=np.int32)
                    global_work_size = np.array([1], dtype=np.int32)
                else:
                    global_work_size = np.array(x.shape, dtype=np.int32)[anchored_axes]
                anchored_axes = cl.array.to_device(Device.GPU.queue, anchored_axes)

                res = np.empty(global_work_size, dtype=np.float32)
                res = cl.array.to_device(Device.GPU.queue, res)
                res_strides = np.array(res.strides, dtype=np.int32) // (res.nbytes // res.size)
                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)

                return global_work_size, None, res, x.data, strides.data, anchored_axes.data, \
                        np.int32(axis), np.int32(x.shape[axis]), np.int32(len(global_work_size)), \
                        res.data, res_strides.data

        def kernel_wrapper(kernel):
            def wrapper(global_work_size, local_work_size, res, *args):
                # call the kernel
                kernel(Device.GPU.queue, global_work_size, local_work_size, *args)
                return res
            return wrapper


class Tensor(object):

    def __init__(self, value):
        self.debug = ""
        self.val = Device.to_device(value)
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

    def _backward(self, dv):
        self.grad += dv
        while self.backward_fxns:
            backward = self.backward_fxns.pop()
            args = self.grad, *self.arguments.pop()
            self.grad = backward(*args)

        assert not type(self.grad) is tuple

    def backward(self):
        # implicit gradient creation
        self._backward(Device.to_device(np.ones(self.shape)))
        
    def operation(expr):
        @functools.wraps(expr)
        def wrapper(self, *args, parsers=(Device.Parser.default,)*2, **kwargs):
            forward, backward = Device.load(expr, *parsers, **kwargs)

            # save intermediate variables for backward pass
            self.arguments.append(args)
            self.backward_fxns.append(backward)

            self.val = forward(*args)

            return self
        return wrapper

    def unary_operation(expr):
        @functools.wraps(expr)
        def wrapper(self, **kwargs):
            return Tensor.operation(expr)(self, self.val, **kwargs)
        return wrapper

    def unary_reduction_operation(expr):
        @functools.wraps(expr)
        def wrapper(self, *args, **kwargs):
            return Tensor.unary_operation(expr)(self, *args, \
                parsers=(Device.Parser.reduction, Device.Parser.binary), **kwargs)
        return wrapper

    def binary_operation(expr):
        def unbroadcast(backward):
            def reduce(grad, shape):
                _shape = (-1,) * abs(len(shape)-len(grad.shape)) + shape
                idx = np.not_equal(grad.shape, _shape)
                axes = *np.arange(grad.ndim)[idx],
                if not axes:
                    return grad
                return Device.load_directly("sum", Device.Parser.reduction, axis=axes)(grad).reshape(shape)
            def wrapper(dv, x, y):
                return *it.starmap(reduce, zip(backward(dv, x, y), (y.shape, x.shape))),
            return wrapper

        def propagate(backward, operand):
            def wrapper(dv, x, y):
                dy, dx = backward(dv, x, y)
                operand._backward(dy)
                return dx
            return wrapper

        def wrapper(self, operand, **kwargs):
            @functools.wraps(expr)
            def expr_wrapper(**kwargs):
                forward, backward = expr(**kwargs)
                # unbroadcast results from backward pass and
                # propagate back on operand
                return forward, propagate(unbroadcast(backward), operand)
            args = self.val, operand.val
            return Tensor.operation(expr_wrapper)(self, *args, \
                parsers=(Device.Parser.binary,)*2, **kwargs)
        return wrapper

    operation.unary = unary_operation
    operation.unary.reduction = unary_reduction_operation
    operation.binary = binary_operation

    # ========== unary ops ==========

    @operation.unary
    def exp():

        def forward(x):
            return np.exp(x)

        def backward(dv, x):
            return dv * np.exp(x)

        return forward, backward

    @operation.unary
    def log():
        def forward(x):
            # x = np.clip(x, 1e-7, 1 - 1e-7)
            return np.log(x)

        def backward(dv, x):
            return dv / x

        return forward, backward

    # ========== reduce ops ==========

    @operation.unary.reduction
    def sum(axis=None, keepdims=False): # TODO: implement this using einsum "ij -> j" will sum along axis=0

        def forward(x):
            return np.sum(x, axis=axis, keepdims=keepdims)
        
        def backward(dv, x):
            if x.ndim > dv.ndim:
                dv = np.expand_dims(dv, -1)
            return np.broadcast_to(dv, x.shape)

        return forward, backward

    @operation.unary
    def max(axis=None, keepdims=False):

        def forward(x):
            return np.max(x, axis=axis, keepdims=keepdims)
        
        def backward(dv, x):
            if axis:
                area = np.prod(np.take(x.shape, axis))
                xr = x.reshape(-1, area)
                idx = xr.argmax(1)
                mask = np.zeros_like(xr)
                np.put_along_axis(mask, idx[:, None], 1, axis=-1)
                mask = mask * dv.reshape(mask.shape[0], 1)
                return mask.reshape(x.shape)

            idx = tuple(np.argwhere(x == x.max())[0])
            mask = np.zeros_like(x)
            mask[idx] = 1
            return mask*dv

        return forward, backward

    # ========== binary ops ==========

    @operation.binary
    def maximum():

        def forward(x, y):
            return np.maximum(x, y)

        def backward(dv, x, y):
            return dv * (x <= y), dv * (x >= y)
        
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
    def add():
        def forward(x, y):
            return x + y

        def backward(dv, x, y):
            return dv, dv

        return forward, backward

    # ========== processing ops ==========

    @operation.binary
    def dot(subscripts='i...j,j...h->i...h'):
        input_subs, output_subs = subscripts.split('->')
        x_subs, y_subs = input_subs.split(',')

        def forward(x, y):
            return np.einsum(subscripts,x,y)
        
        def backward(dv, x, y):
            assert set(x_subs) <= (set(output_subs) | set(y_subs)), "Tiling (?) not implemented"
            assert set(y_subs) <= (set(output_subs) | set(x_subs))

            dx = np.einsum(f"{output_subs},{y_subs}->{x_subs}", dv, y)
            dy = np.einsum(f"{output_subs},{x_subs}->{y_subs}", dv, x)
            return dy, dx

        return forward, backward

    @operation.unary
    def window_view(kernel_size=(2,2), stride=1):
        ws = None # windows

        def forward(x):
            nonlocal ws
            N, cin, *in_shape, kdims = *x.shape, len(kernel_size)

            # get window shape and strides
            truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
            out_shape = N, cin, *truncated_out_shape, *kernel_size
            out_strides = *x.strides[:2], *(xs*stride for xs in x.strides[-kdims:]), *x.strides[-kdims:]

            # return window view
            return (ws := np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides))
        
        def backward(dv, x):
            nonlocal ws
            assert dv.shape == ws.shape
            ws[:] = 0
            # can this be vectorized?
            for i in np.ndindex(ws.shape):
                # accumulate gradient
                ws[i] += dv[i]

            return x

        return forward, backward

    # ========== composite ops ==========

    def relu(self):
        return self.maximum(Tensor(0))

    def div(self, x):
        assert isinstance(x, type(self))
        return self.mul(x.pow(Tensor(-1.0)))

    def mul(self, x):
        return self.dot(x, subscripts="...,...->...")

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
        return self.window_view(kernel_size=w.shape[-2:], stride=stride)\
            .dot(w, subscripts='abcdef,gbef->agcd')

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
        self.debug += " fork "

        dv_fork = None

        def fork_backward(dv):
            nonlocal dv_fork
            dv_fork = dv
            return dv

        fork = Tensor(self.val)
        fork.backward_fxns.append(fork_backward)
        fork.arguments.append(tuple()) # empty args

        def parent_backward(dv):
            nonlocal dv_fork
            assert dv_fork is not None
            return dv + dv_fork # accumulate parent gradient

        self.backward_fxns.append(parent_backward)
        self.arguments.append(tuple()) # empty args

        fork.debug += " join "

        return fork