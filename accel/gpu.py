import numpy as np

import pyopencl as cl
import pyopencl.array

import functools
import itertools as it

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
            func = getattr(GPU, func.__name__)
            return func(*inputs, **kwargs)

        def __sub__(x, y): return x + y * -1
        def __rsub__(x, y): return y + x * -1

        def __truediv__(x, y): return x * y ** -1
        def __rtruediv__(x, y): return y * x ** -1

        def __repr__(self):
            return f"GPUArray({np.array2string(self.get(), 88, 4, True, ', ', 'GPUArray(', suffix=')')})"

        def get(self):
            res = np.empty(self.data.size // self.dtype.itemsize, self.dtype)
            cl.enqueue_copy(GPU.queue, res, self.data)
            if not self.shape:
                return res
            return np.lib.stride_tricks.as_strided(res, self.shape, self.strides)

        def reshape(self, *shape):
            if -1 in shape:
                shape = tuple(x if x > 0 else 
                        int(abs(np.prod(self.shape) / np.prod(shape)))
                        for x in shape)
            result = GPU.Array(shape, self.dtype)
            result.data = self.data
            return result

        def squeeze(self):
            shape = tuple(np.compress(np.array(self.shape) > 1, self.shape))
            result = GPU.Array(shape)
            result.data = self.data
            return result

        def transpose(self, *order):
            shape = tuple(np.take(self.shape, order))
            result = self.__class__(shape)
            result.strides = tuple(np.take(self.strides, order))
            result.data = self.data
            return result

        def copy(self):
            return GPU.copy(self) # only works for np.int32 type atm

    def __init__(self):
        # initialize opencl
        GPU.ctx = cl.create_some_context()
        GPU.queue = cl.CommandQueue(self.ctx)

        prg = cl.Program(GPU.ctx, open('./accel/gpu_ops.cl').read()).build()
        for kernel in prg.all_kernels():
            tokens = kernel.function_name.split("__")
            assert len(tokens) == 2
            name, parser = tokens
            parser = getattr(self.Parser, parser)
            wrapped_gpu_op = self.Parser.wrapper(parser, functools.partial(kernel, self.queue))
            setattr(GPU, name, wrapped_gpu_op)

    def to_cpu(x):
        return x.get()

    def reshape(x, shape):
        if np.isscalar(shape):
            shape = (shape,)
        if x.base:
            return GPU.broadcast_to(x, shape)
        return x.reshape(*shape)

    def broadcast_to(x, shape):
        if x.base:
            x = x.base

        # set strides to 0 for all singleton dimensions
        strides = np.where(np.equal(x.shape, 1), 0, x.strides)
        # add empty trailing strides if needed
        strides = np.append(strides, np.array([0]*abs(x.ndim - len(shape)), int))
        
        arr = GPU.Array(shape)
        arr.data = x.data
        arr.strides = tuple(strides)
        arr.base = x
        return arr

    def as_strided(x, shape, strides):
        arr = GPU.Array(shape)
        arr.data = x.data
        arr.shape = shape
        arr.strides = strides
        arr.dtype = x.dtype
        arr.nbytes = x.nbytes
        return arr

    def array(arr, dtype=np.float32, ndmin=1, **kwargs):
        if isinstance(arr, GPU.Array):
            return arr
        arr = np.array(arr, copy=False, dtype=dtype, ndmin=ndmin, **kwargs)
        if arr.size:
            data = cl.Buffer(GPU.ctx, cl.mem_flags.READ_WRITE |
                                cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)
        else:
            data = None
        return GPU.Array(arr.shape, dtype, data)

    def empty(shape, dtype=np.float32):
        arr = GPU.Array(shape, dtype)
        arr.data = cl.Buffer(GPU.ctx, cl.mem_flags.READ_WRITE, arr.nbytes)
        return arr

    def arange(n):
        return GPU.array(np.arange(n), dtype=np.int32)

    class Parser(object):
        def wrapper(parser, kernel):
            def _wrapper(*args, **kwargs):
                args = tuple(x if isinstance(x, (str, GPU.Array)) 
                            else GPU.array(x) for x in args)
                return parser(kernel, *args, **kwargs)
            return _wrapper
        
        def elementwise(kernel, *args, **kwargs):
            # allocate output buffer on device
            res = GPU.empty(args[0].shape)
            kernel([args[0].size], None, *(a.data for a in (*args, res)))
            return res

        def broadcast(kernel, *args):
            res_shape = np.broadcast_shapes(*(x.shape for x in args))
            res = GPU.empty(res_shape, dtype=args[0].dtype)
            res_strides = np.arange(np.prod(res_shape), dtype=np.int32)

            args_strides = tuple(
                np.broadcast_to(
                np.lib.stride_tricks.as_strided(
                np.arange(np.prod(x.shape), dtype=np.int32), x.shape, x.strides),
                res_shape).flatten()
                for x in args)
            
            # convert to opencl
            args = tuple(it.chain(*zip((*args, res), (cl.array.to_device(GPU.queue, x) 
                                                        for x in (*args_strides, res_strides)))))

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
                return GPU.multiply(x, y)

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

            res = GPU.empty(res_shape)
            res_strides = np.arange(np.prod(res_shape), dtype=np.int32)

            # convert to opencl
            reduced_axis_size = np.int32(reduced_axis_size)
            x_strides = cl.array.to_device(GPU.queue, xstrides)
            y_strides = cl.array.to_device(GPU.queue, ystrides)

            reduced_axis_stride_x = cl.array.to_device(GPU.queue, reduced_axes_stride_x)
            reduced_axis_stride_y = cl.array.to_device(GPU.queue, reduced_axes_stride_y)

            res_strides = cl.array.to_device(GPU.queue, res_strides)

            # call kernel
            kernel([np.prod(res_shape)], None, x.data, y.data, x_strides.data, y_strides.data, \
                reduced_axis_stride_x.data, reduced_axis_stride_y.data, reduced_axis_size, res.data, res_strides.data)

            return res

        def bincount(kernel, x, w, minlength):
            res_np = np.zeros(minlength)
            res = GPU.array(res_np)
            # res_strides = np.arange(np.prod(res_shape), dtype=np.int32)
            kernel([x.size], None, x.data, w.data, res.data)
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
            result = GPU.empty(tuple(result_shape))
            result_strides = np.arange(np.prod(result_shape), dtype=np.int32)

            # convert to opencl
            reduced_axes_stride = cl.array.to_device(GPU.queue, reduced_axes_stride)
            xstrides = cl.array.to_device(GPU.queue, xstrides)
            reduced_axis_size = np.int32(np.prod(reduced_shape))
            result_strides = cl.array.to_device(GPU.queue, result_strides)

            args = x.data, xstrides.data, reduced_axes_stride.data, reduced_axis_size, \
                    result.data, result_strides.data

            kernel([np.prod(result_shape)], None, *args)

            return result