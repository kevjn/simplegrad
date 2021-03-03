import numpy as np
from simplegrad.tensor import Device, Tensor

np.random.seed(1337)
Device.load_device(Device.GPU)

def equal_sum_over_axis(arr, *, axis):
    res_cpu = np.sum(arr, axis=axis)
    res_gpu = Tensor(arr).sum(axis=axis).val
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def equal_sum_over_all_axes(arr):
    res_cpu = np.sum(arr)
    a = Tensor(arr)
    for _ in range(arr.ndim):
        res_gpu = a.sum(axis=0)
    res_gpu = res_gpu.val
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def test_sum_reduction_kernel():
    a = np.random.randn(10,100,60,5).astype(np.float32)

    for i in range(a.ndim):
        assert equal_sum_over_axis(a, axis=i)

def test_sum_all_axes():
    a = np.random.randn(5,10,20,25).astype(np.float32)
    assert equal_sum_over_all_axes(a)