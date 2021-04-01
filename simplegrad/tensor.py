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
        return *(Device.operation_wrapper(op, parser) for op, parser in zip(cpu_expr(**kwargs), parsers)),

    def load_directly(name, parser, backward=False, **kwargs):
        # get unwrapped function
        f = getattr(Tensor, name).__wrapped__
        return f(**kwargs)[backward]

    def to_device(x):
        return np.array(x, dtype=np.float32)

    def operation_wrapper(op, parser, **kwargs):
        def wrapper(*args):
            return parser(op, *args, **kwargs)
        return wrapper

    def drop_output_wrapper(parser):
        def wrapper(*args):
            *args, out = args
            return parser(*args)
        return wrapper

    class CPU:
        class Parser(object):
            def default(op, *args, **kwargs):
                return op(*args)

        Parser.reduction = Parser.binary = Parser.binary_reduction = Parser.default
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
                ops[name][direction != "forward"] = functools.partial(kernel, Device.GPU.queue)

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

            def default(kernel, *args, **kwargs):
                # allocate output buffer on device
                res = cl.array.empty_like(args[0])
                kernel([args[0].size], None, *(a.data for a in (*args, res)))
                return res

            def binary(kernel, x, y, **kwargs):
                res = cl.array.empty(Device.GPU.queue, np.broadcast_shapes(x.shape, y.shape), np.float32)
                res_strides = cl.array.to_device(Device.GPU.queue, np.array(res.strides, dtype=np.int32) // 4)

                xstrides = (np.equal(np.pad(x.shape, (res.ndim-x.ndim, 0)), res.shape) * \
                            np.pad(x.strides, (res.ndim-x.ndim, 0)) // 4).astype(np.int32)
                xstrides = cl.array.to_device(Device.GPU.queue, xstrides)

                ystrides = (np.equal(np.pad(y.shape, (res.ndim-y.ndim, 0)), res.shape) * \
                            np.pad(y.strides, (res.ndim-y.ndim, 0)) // 4).astype(np.int32)
                ystrides = cl.array.to_device(Device.GPU.queue, ystrides)

                args = x.data, xstrides.data, y.data, ystrides.data, \
                        res.data, np.int32(res.ndim), res_strides.data
                
                kernel(res.shape, None, *args)
                return res

            def binary_reduction(kernel, x, y, subscripts=None):
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

                reduced_subscripts = set(x_subs) & set(y_subs) - set(out_subs)

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
                    return Device.GPU.load_directly("mul", Device.GPU.Parser.binary)(x, y)

                assert len(reduced_subscripts) == 1, "reduction over multiple axis not implemented yet"
                reduced_subscript = reduced_subscripts.pop()

                # reduced dimension in operands
                reduced_axis_x = x_subs.index(reduced_subscript)
                reduced_axis_y = y_subs.index(reduced_subscript)

                # corresponding stride
                reduced_axis_stride_x = xstrides[reduced_axis_x]
                reduced_axis_stride_y = ystrides[reduced_axis_y]

                assert x.shape[xorder[reduced_axis_x]] == y.shape[yorder[reduced_axis_y]]
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
                    ystrides = np.insert(ystrides, axis, 0)

                broadcasted_subs_x = len(x_subs) < len(y_subs) and set(y_subs) - set(x_subs)
                if broadcasted_subs_x:
                    axis = np.take(yorder, [y_subs.index(s) for s in broadcasted_subs_x])
                    xstrides = np.insert(xstrides, axis, 0)

                res = cl.array.empty(Device.GPU.queue, res_shape, np.float32)

                max_ndim, max_shape = max((x.ndim, x.shape), (y.ndim, y.shape), key=operator.itemgetter(1))

                res_strides = not res.ndim < max_ndim and res.strides or \
                    tuple(np.insert(res.strides, out_subs.index(reduced_subscript), 0))
                res_strides = np.array(res_strides, dtype=np.int32) // 4

                # convert to opencl
                reduced_axis_size = np.int32(reduced_axis_size)
                x_strides = cl.array.to_device(Device.GPU.queue, xstrides)
                y_strides = cl.array.to_device(Device.GPU.queue, ystrides)

                reduced_axis_stride_x = np.int32(reduced_axis_stride_x)
                reduced_axis_stride_y = np.int32(reduced_axis_stride_y)

                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)

                # call kernel
                kernel(max_shape, None, x.data, y.data, x_strides.data, y_strides.data, \
                    reduced_axis_stride_x, reduced_axis_stride_y, reduced_axis_size, res.data, res_strides.data)

                return res

            def reduction(kernel, x, axis=None, **kwargs):
                if axis is None and (axis := tuple(range(x.ndim))) or type(axis) is tuple:
                    for ax in sorted(axis, reverse=True):
                        x = Device.GPU.Parser.reduction(kernel, x, axis=ax)
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

                res = cl.array.empty(Device.GPU.queue, tuple(global_work_size), np.float32)
                res_strides = np.array(res.strides, dtype=np.int32) // (res.nbytes // res.size)
                res_strides = cl.array.to_device(Device.GPU.queue, res_strides)
                
                args = x.data, strides.data, anchored_axes.data, \
                       np.int32(axis), np.int32(x.shape[axis]), res.data, res_strides.data

                kernel(global_work_size, None, *args)
                return res


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
        
    def operation(expr, *, parsers):
        @functools.wraps(expr)
        def wrapper(self, *args, **kwargs):
            forward, backward = Device.load(expr, *parsers, **kwargs)

            self.val = forward(*args)

            # save intermediate variables and output for backward pass
            self.arguments.append((*args, self.val))
            self.backward_fxns.append(backward)

            return self
        return wrapper

    def unary_operation(forward_parser=Device.Parser.default,
                        backward_parser=Device.drop_output_wrapper(Device.Parser.default)):
        def decorator(expr):
            @functools.wraps(expr)
            def wrapper(self, **kwargs):
                return Tensor.operation(expr, parsers=(forward_parser, backward_parser))(self, self.val, **kwargs)
            return wrapper
        return decorator

    def binary_operation(forward_parser=Device.Parser.binary, 
                         backward_parser=Device.drop_output_wrapper(Device.Parser.binary)):
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

        def decorator(expr):
            @functools.wraps(expr)
            def wrapper(self, operand, **kwargs):
                @functools.wraps(expr)
                def expr_wrapper(**kwargs):
                    forward, backward = expr(**kwargs)
                    # unbroadcast results from backward pass and
                    # propagate back on operand
                    return forward, propagate(unbroadcast(backward), operand)
                args = self.val, operand.val
                return Tensor.operation(expr_wrapper, parsers=(forward_parser, backward_parser))(self, *args, **kwargs)
            return wrapper
        return decorator

    operation.unary = unary_operation
    operation.binary = binary_operation

    # ========== unary ops ==========

    @operation.unary()
    def relu():

        def forward(x):
            return np.maximum(x, 0)

        def backward(dv, x):
            return dv * (x >= 0)
        
        return forward, backward

    @operation.unary(backward_parser=Device.Parser.binary)
    def exp():

        def forward(x):
            return np.exp(x)

        def backward(dv, x, out):
            return dv * out

        return forward, backward

    @operation.unary()
    def log():
        def forward(x):
            return np.log(x)

        def backward(dv, x):
            return dv / x

        return forward, backward

    # ========== reduce ops ==========

    @operation.unary(forward_parser=Device.Parser.reduction)
    def sum(axis=None, keepdims=False):

        def forward(x):
            return np.sum(x, axis=axis, keepdims=keepdims)
        
        def backward(dv, x):
            if x.ndim > dv.ndim:
                dv = np.expand_dims(dv, -1)
            return np.broadcast_to(dv, x.shape)

        return forward, backward

    @operation.unary(forward_parser=Device.Parser.reduction, backward_parser=Device.Parser.binary)
    def max(axis=None, keepdims=False):

        def forward(x):
            return np.max(x, axis=axis, keepdims=keepdims)
        
        def backward(dv, x, out):
            if axis:
                area = np.prod(np.take(x.shape, axis))
                xr = x.reshape(-1, area)
                idx = xr.argmax(1)
                mask = np.zeros_like(xr)
                np.put_along_axis(mask, idx[:, None], 1, axis=-1)
                mask = mask * dv.reshape(mask.shape[0], 1)
                return mask.reshape(x.shape)

            mask = np.zeros_like(x)
            mask[x == out] = dv
            return mask

        return forward, backward

    # ========== binary ops ==========

    @operation.binary()
    def pow():

        def forward(x, y):
            return x ** y

        def backward(dv, x, y):
            return dv * x ** y * np.log(x), \
                   dv * y * x ** (y-1.0)

        return forward, backward

    @operation.binary()
    def add():
        def forward(x, y):
            return x + y

        def backward(dv, x, y):
            return dv, dv

        return forward, backward

    # ========== processing ops ==========

    @operation.binary(forward_parser=Device.Parser.binary_reduction)
    def dot(subscripts='i...j,j...k->i...k'):
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

    @operation.unary(backward_parser=Device.Parser.binary)
    def window_view(kernel_size=(2,2), stride=1):

        def forward(x):
            N, cin, *in_shape, kdims = *x.shape, len(kernel_size)

            # get window shape and strides
            truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
            out_shape = N, cin, *truncated_out_shape, *kernel_size
            out_strides = *x.strides[:2], *(xs*stride for xs in x.strides[-kdims:]), *x.strides[-kdims:]

            # return window view
            return np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides)
        
        def backward(dv, x, out):
            assert dv.shape == out.shape
            out[:] = 0
            # can this be vectorized?
            for i in np.ndindex(out.shape):
                # accumulate gradient
                out[i] += dv[i]

            return x

        return forward, backward

    # ========== composite ops ==========

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