import numpy as np
from simplegrad.tensor import Device, Tensor

np.random.seed(1337)
Device.load_device(Device.GPU)

def equal_sum_over_axis(arr, *, axis):
    res_cpu = np.sum(arr, axis=axis)
    res_gpu = Tensor(arr).sum(axis=axis).val
    return np.allclose(res_cpu, res_gpu, rtol=1e-04, atol=1e-07)

def test_sum_reduction_kernel():
    a = np.random.randn(10,100,60,5).astype(np.float32)

    for i in range(a.ndim):
        assert equal_sum_over_axis(a, axis=i)