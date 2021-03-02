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
        Device.to_device = device.to_device

    def load(cpu_expr, **kwargs):
        # use cpu forward/backward fxn as default
        return cpu_expr(**kwargs) 

    def to_device(x):
        return np.array(x)

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
                ops[name][direction != "forward"] = Device.GPU.opencl_parser(kernel)

            # freeze dictionary values
            Device.GPU.ops = defaultdict(lambda: (None, None), \
                {name : tuple(v) for name, v in ops.items()})

        def load(cpu_expr, **kwargs):
            # fall back on cpu if no implementation exists, TODO: memoize this
            return *(a(**kwargs) if a else b for a,b in zip(Device.GPU.ops[cpu_expr.__name__], cpu_expr(**kwargs))),

        def to_device(x: np.ndarray):
            return cl.array.to_device(Device.GPU.queue, x.astype(np.float32))

        def opencl_parser(kernel):
            def kwargs_wrapper(axis=None, **kwargs):
                def wrapper(*args):
                    if kernel.function_name == 'sum_forward':
                        # reduction kernel
                        assert len(args) == 1, "binary reduction not implemented yet"
                        assert axis is not None

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

                        kernel(Device.GPU.queue, global_work_size, None, x.data, strides.data, anchored_axes.data, 
                               np.int32(axis), np.int32(x.shape[axis]), np.int32(len(global_work_size)), res.data, res_strides.data)

                        return res

                    # allocate output buffer on device
                    res = cl.array.empty_like(args[0])

                    # call the kernel, leave the local work size (None) to the implementation
                    kernel(Device.GPU.queue, [args[0].size], None, *(a.data for a in args), res.data)

                    return res
                return wrapper
            return kwargs_wrapper


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
        self._backward(np.ones(self.shape))
        
    def operation(expr):
        def wrapper(self, *args, **kwargs):
            forward, backward = Device.load(expr, **kwargs)

            # save intermediate variables for backward pass
            self.arguments.append(args)
            self.backward_fxns.append(backward)

            self.val = forward(*args)

            return self
        return wrapper

    def unary_operation(expr):
        def wrapper(self, *args, **kwargs):
            assert not args, "unary operations can't have positional arguments"
            args = self.val, 

            return Tensor.operation(expr)(self, *args, **kwargs)
        return wrapper

    def binary_operation(expr):
        def unbroadcast(backward):
            def wrapper(dv, x, y):
                def generator():
                    for grad, shape in zip(backward(dv, x, y), (y.shape, x.shape)):
                        if shape < grad.shape:
                            fill = shape and shape[-1] or None
                            axis = tuple(i for i, (a,b) in enumerate(it.zip_longest
                                        (grad.shape, shape, fillvalue=fill)) if a!=b)
                            yield grad.sum(axis=axis).reshape(shape)
                        else:
                            yield grad
                
                return *generator(),
            return wrapper

        def propagate(backward, operand):
            def wrapper(*args):
                dy, dx = backward(*args)
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
            return Tensor.operation(expr_wrapper)(self, *args, **kwargs)
        return wrapper

    def abstract_operation(expr):
        def wrapper(self, fxn, *args, **kwargs):
            derived_expr = fxn.__func__.__closure__[0].cell_contents
            base_expr = functools.partial(expr, derived_expr, **kwargs)
            # swap wrapper
            fxn.__func__.__closure__[0].cell_contents = base_expr
            # call correct operation type
            fxn(*args)
            return self
        return wrapper

    operation.unary = unary_operation
    operation.binary = binary_operation
    operation.abstract = abstract_operation

    # ========== unary ops ==========

    @operation.unary
    def relu():

        def forward(x):
            return np.maximum(x, 0)

        def backward(dv, x):
            return dv * (x >= 0)
        
        return forward, backward

    # ========== reduce ops ==========

    @operation.unary
    def sum(**np_kwargs):

        def forward(x):
            return np.sum(x, **np_kwargs)
        
        def backward(dv, x):
            if x.ndim > dv.ndim:
                dv = np.expand_dims(dv, -1)
            return np.broadcast_to(dv, x.shape)

        return forward, backward
    
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

    @operation.unary
    def max(**kwargs):

        def forward(x):
            return np.max(x, **kwargs)
        
        def backward(dv, x):
            idx = tuple(np.argwhere(x == x.max(**kwargs))[0])
            mask = np.zeros_like(x)
            mask[idx] = 1
            return mask*dv

        return forward, backward

    # ========== binary ops ==========

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
    def dot(subscripts='i...j,j...h->i...h'):
        input_subs, output_subs = subscripts.split('->')
        x_subs, y_subs = input_subs.split(',')

        def forward(x, y):
            return np.einsum(subscripts,x,y)
        
        def backward(dv, x, y):
            dx = np.einsum(f"{output_subs},{y_subs}->{x_subs}", dv, y)
            dy = np.einsum(f"{output_subs},{x_subs}->{y_subs}", dv, x)
            return dy, dx

        return forward, backward

    # ========== abstract ops ==========

    @operation.abstract
    def sliding_window(kernel_fxn, kernel_size=(2,2), stride=1, **kwargs):
        kernel_forward, kernel_backward = kernel_fxn(**kwargs)
        ws = None # windows

        def forward(x, operand=None):
            nonlocal ws
            N, cin, *in_shape, kdims = *x.shape, len(kernel_size)

            # get window shape and strides
            truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
            out_shape = N, cin, *truncated_out_shape, *kernel_size
            out_strides = *x.strides[:2], *(xs*stride for xs in x.strides[-kdims:]), *x.strides[-kdims:]

            # get windows
            ws = np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides)

            if operand is not None:
                cout = operand.shape[0]
                # init truncated output and set each element
                out = np.zeros((N, cout, *truncated_out_shape))
                for N, cout, *i in np.ndindex(out.shape):
                    out[(N, cout, *i)] = kernel_forward(ws[(N, slice(None), *i)], operand[cout])
            else:
                cout = cin
                out = np.zeros((N, cout, *truncated_out_shape))
                for N, cout, *i in np.ndindex(out.shape):
                    out[(N, cout, *i)] = kernel_forward(ws[(N, cout, *i)])

            return out
        
        def backward(dv, x, operand=None):
            nonlocal ws
            out_shape = dv.shape

            x_ws = ws
            dx = np.zeros_like(x, dtype='float64')
            dx_ws = np.lib.stride_tricks.as_strided(dx, shape=x_ws.shape, strides=x_ws.strides)

            if operand is not None:
                dw = np.zeros_like(operand, dtype='float64')

                for N, cout, *i in np.ndindex(*out_shape):
                    grad = kernel_backward(dv[(N, cout, *i)], \
                            x_ws[(N, slice(None), *i)], operand[cout])
                    # accumulate gradients
                    for tot_grad, grad in zip((dw[cout], dx_ws[(N, slice(None), *i)]), grad):
                        tot_grad += grad

                return dw, dx
            else:
                for N, cout, *i in np.ndindex(*out_shape):
                    grad = kernel_backward(dv[(N, cout, *i)], x_ws[(N, cout, *i)])
                    # accumulate gradient
                    dx_ws[(N, cout, *i)] += grad 

                return dx

        return forward, backward

    # ========== composite ops ==========

    def div(self, x):
        assert isinstance(x, type(self))
        return self.mul(x.pow(Tensor(-1.0)))

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

    def mean(self):
        a = Tensor(self.shape[-1])
        return self.sum(axis=-1, keepdims=True).div(a)

    def conv2d(self, w, padding=0, stride=1):
        return self.sliding_window(self.dot, w, \
            subscripts='...ijk,...ijk->...', kernel_size=w.shape[-2:], stride=stride)

    def maxpool2d(self, kernel_size = (3,3), padding=0, stride=1):
        return self.sliding_window(self.max, kernel_size=kernel_size, stride=stride)

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