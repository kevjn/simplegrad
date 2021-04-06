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
        Device.to_device = device.to_device
        Device.Parser = device.Parser

    def load(name):
        # get unwrapped forward function
        return getattr(Tensor, name).__closure__[1].cell_contents

    def to_device(x):
        return np.array(x, dtype=np.float32)

    def to_cpu(x):
        return x

    class CPU:
        pass

    class GPU:
        def __init__(self):
            # initialize opencl
            Device.GPU.ctx = cl.create_some_context()
            Device.GPU.queue = cl.CommandQueue(self.ctx)

            prg = cl.Program(Device.GPU.ctx, open('./accelerators/gpu_ops.cl').read()).build()
            for kernel in prg.all_kernels():
                tokens = kernel.function_name.split("_")
                assert len(tokens) == 3
                name, direction, parser = tokens
                
                # set appropriate interface to kernel
                parser = getattr(self.Parser, parser)
                wrapped_gpu_op = functools.partial(parser, functools.partial(kernel, self.queue))

                if direction == 'extra': # extra operation
                    setattr(Device.GPU, name, wrapped_gpu_op)
                    continue

                # reassign closure with decorated gpu op
                op = getattr(Tensor, name)
                op.__closure__[direction == 'forward'].cell_contents = wrapped_gpu_op

            # sum backward needs an additional wrapper
            _sum = Tensor.sum.__closure__[0].cell_contents
            wrapped_gpu_op = functools.partial(Device.GPU.Parser.sum_backward_wrapper, _sum)
            Tensor.sum.__closure__[0].cell_contents = wrapped_gpu_op

            # einsum backward uses forward pass and needs an additional wrapper
            Tensor.dot.__closure__[0].cell_contents = Tensor.dot.__closure__[1].cell_contents
            _dot = Tensor.dot.__closure__[0].cell_contents
            wrapped_gpu_op = functools.partial(Device.GPU.Parser.einsum_backward_wrapper, _dot)
            Tensor.dot.__closure__[0].cell_contents = wrapped_gpu_op

            # assume pow backward is commutative
            _pow = Tensor.pow.__closure__[0].cell_contents
            wrapped_gpu_op = functools.partial(Device.GPU.Parser.backward_wrapper, _pow)
            Tensor.pow.__closure__[0].cell_contents = wrapped_gpu_op

        def to_device(x):
            return cl.array.to_device(Device.GPU.queue, x.astype(np.float32)) \
                if isinstance(x, np.ndarray) else x.copy() \
                if isinstance(x, cl.array.Array) else \
                cl.array.to_device(Device.GPU.queue, np.array([x], dtype=np.float32))

        def to_cpu(x):
            return x.get()

        class Parser(object):

            def backward_wrapper(wrapped_op, *args, **kwargs):
                dv = wrapped_op(*args, **kwargs)
                return dv, dv

            def sum_backward_wrapper(wrapped_op, dv, x, axis=None, keepdims=False):
                if x.ndim > dv.ndim:
                    broadcasted_shape = np.array((slice(None),) * x.ndim)
                    np.put(broadcasted_shape, axis, None)
                    dv = dv[tuple(broadcasted_shape)]
                return wrapped_op(dv, x, axis=axis, keepdims=keepdims)

            def einsum_backward_wrapper(wrapped_op, *args, subscripts="ij,jk->ik"):
                dv, x, y = args
                input_subs, output_subs = subscripts.split('->')
                x_subs, y_subs = input_subs.split(',')
                dx = wrapped_op(dv, y, subscripts=f"{output_subs},{y_subs}->{x_subs}")
                dy = wrapped_op(dv, x, subscripts=f"{output_subs},{x_subs}->{y_subs}")

                return dy, dx

            def default(kernel, *args, **kwargs):
                # allocate output buffer on device
                res = cl.array.empty_like(args[0])
                kernel([args[0].size], None, *(a.data for a in (*args, res)))
                return res

            def binary(kernel, *args, **kwargs):
                res_shape = np.broadcast_shapes(*(x.shape for x in args))
                res = cl.array.empty(Device.GPU.queue, res_shape, np.float32)
                res_strides = cl.array.to_device(Device.GPU.queue, np.array(res.strides, dtype=np.int32) // 4)

                strides = ((cl.array.to_device(Device.GPU.queue, 
                            (np.equal(np.pad(x.shape, (res.ndim-x.ndim, 0)), res.shape) * 
                            np.pad(x.strides, (res.ndim-x.ndim, 0)) // 4).astype(np.int32)).data)
                            for x in args)

                args = it.chain(*zip((a.data for a in args), strides), (res.data, np.int32(res.ndim), res_strides.data))

                kernel(res.shape, None, *args)
                return res

            def binaryreduction(kernel, x, y, subscripts):
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
                    return Device.GPU.mul(x, y)

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

                res = cl.array.empty(Device.GPU.queue, res_shape, np.float32)

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

            def reduction(kernel, x, axis=None, keepdims=False, **kwargs):
                axis = Device.Parser.axis(x, axis)

                if type(axis) is tuple:
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

                res = cl.array.empty(Device.GPU.queue, (*global_work_size, *[1] * keepdims * (x.ndim - global_work_size.size)), np.float32)
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
            *args, kwargs = self.grad, *self.arguments.pop()
            self.grad = backward(*args, **kwargs)

        assert not type(self.grad) is tuple

    def backward(self):
        # implicit gradient creation
        self._backward(Device.to_device(np.ones(self.shape)))
        
    def operation(forward, backward, *, use_output):
        @functools.wraps(forward)
        def wrapper(self, *args, **kwargs):

            self.val = forward(*args, **kwargs)

            # save intermediate variables and (output) for backward pass
            self.arguments.append((*args, *(self.val,) * use_output, kwargs))
            self.backward_fxns.append(backward)

            return self
        return wrapper

    def unary_operation(use_output=False):
        def decorator(forward, backward):
            @functools.wraps(forward)
            def wrapper(self, **kwargs):
                return Tensor.operation(forward, backward, use_output=use_output)(self, self.val, **kwargs)
            return wrapper
        return decorator

    def binary_operation():
        def unbroadcast(backward):
            def reduce(grad, shape):
                _shape = (-1,) * abs(len(shape)-len(grad.shape)) + shape
                idx = np.not_equal(grad.shape, _shape)
                axes = *np.arange(grad.ndim)[idx],
                if not axes:
                    return grad
                return Device.load("sum")(grad, axis=axes).reshape(shape)
            @functools.wraps(backward)
            def wrapper(*args, shapes, **kwargs):
                return *it.starmap(reduce, zip(backward(*args, **kwargs), shapes)),
            return wrapper

        def propagate(backward, operand):
            @functools.wraps(backward)
            def wrapper(*args, **kwargs):
                x, y = args[1:3]
                dy, dx = backward(*args, shapes=(y.shape, x.shape), **kwargs)
                operand._backward(dy)
                return dx
            return wrapper

        def decorator(forward, backward):
            @functools.wraps(forward)
            def wrapper(self, operand, **kwargs):
                args = self.val, operand.val
                return Tensor.operation(forward, propagate(unbroadcast(backward), operand), use_output=False)(self, *args, **kwargs)
            return wrapper
        return decorator

    operation.unary = unary_operation 
    operation.binary = binary_operation

    # ========== unary ops ==========

    def relu_forward(x): 
        return np.maximum(x, 0)

    def relu_backward(dv, x): 
        return dv * (x >= 0)

    relu = operation.unary()(relu_forward, relu_backward)

    def exp_forward(x):
        return np.exp(x)

    def exp_backward(dv, x, out):
        return dv * out

    exp = operation.unary(use_output=True)(exp_forward, exp_backward)
    
    def log_forward(x):
        return np.log(x)

    def log_backward(dv, x):
        return dv / x
    
    log = operation.unary()(log_forward, log_backward)

    # ========== reduce ops ==========

    def sum_forward(x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def sum_backward(dv, x, **kwargs):
        if x.ndim > dv.ndim:
            dv = np.expand_dims(dv, -1)
        return np.broadcast_to(dv, x.shape)

    sum = operation.unary()(sum_forward, sum_backward)

    def max_forward(x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)
    
    def max_backward(dv, x, out, axis=None, **kwargs):
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

    max = operation.unary(use_output=True)(max_forward, max_backward)

    # ========== binary ops ==========

    def pow_forward(x, y):
        return x ** y

    def pow_backward(dv, x, y):
        return dv * x ** y * np.log(x), \
                dv * y * x ** (y-1.0)

    pow = operation.binary()(pow_forward, pow_backward)

    def add_forward(x, y):
        return x + y

    def add_backward(dv, x, y):
        return dv, dv

    add = operation.binary()(add_forward, add_backward)

    # ========== processing ops ==========

    def dot_forward(x, y, subscripts='i...j,j...k->i...k'):
        return np.einsum(subscripts,x,y)
    
    def dot_backward(dv, x, y, subscripts='i...j,j...k->i...k'):
        input_subs, output_subs = subscripts.split('->')
        x_subs, y_subs = input_subs.split(',')
        reduced_subscripts_x = set(x_subs) - set(output_subs + y_subs)
        x_subs_non_reduced = "".join(filter(lambda x: x not in reduced_subscripts_x, x_subs))
        dx = np.einsum(f"{output_subs},{y_subs}->{x_subs_non_reduced}", dv, y)

        reduced_subscripts_y = set(y_subs) - set(output_subs + x_subs)
        y_subs_non_reduced = "".join(filter(lambda y: y not in reduced_subscripts_y, y_subs))
        dy = np.einsum(f"{output_subs},{x_subs}->{y_subs_non_reduced}", dv, x)

        if reduced_subscripts_x:
            raise NotImplementedError # Tiling not implemented

        if reduced_subscripts_y:
            raise NotImplementedError # Tiling not implemented

        return dy, dx

    dot = operation.binary()(dot_forward, dot_backward)

    def window_view_forward(x, kernel_size=(2,2), stride=1):
        N, cin, *in_shape, kdims = *x.shape, len(kernel_size)

        # get window shape and strides
        truncated_out_shape = *((xin - kin) // 1 + 1 for xin, kin in zip(in_shape, kernel_size)),
        out_shape = N, cin, *truncated_out_shape, *kernel_size
        out_strides = *x.strides[:2], *(xs*stride for xs in x.strides[-kdims:]), *x.strides[-kdims:]

        # return window view
        return np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides)
    
    def window_view_backward(dv, x, out, **kwargs):
        assert dv.shape == out.shape
        out[:] = 0
        # can this be vectorized?
        for i in np.ndindex(out.shape):
            # accumulate gradient
            out[i] += dv[i]

        return x

    window_view = operation.unary(use_output=True)(window_view_forward, window_view_backward)

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
        fork.arguments.append((dict(),)) # empty args

        def parent_backward(dv):
            nonlocal dv_fork
            assert dv_fork is not None
            return dv + dv_fork # accumulate parent gradient

        self.backward_fxns.append(parent_backward)
        self.arguments.append((dict(),)) # empty args

        fork.debug += " join "

        return fork