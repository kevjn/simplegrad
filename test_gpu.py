import numpy as np
from simplegrad.tensor import Device, Tensor
import torch

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

def equal_add(a: np.ndarray, b: np.ndarray):
    tensor_pytorch = torch.tensor(a).add(torch.tensor(b))
    tensor_simplegrad = Tensor(a).add(Tensor(b))
    return np.allclose(tensor_simplegrad.val.get(), tensor_pytorch.data.numpy())

def test_sum_reduction_kernel():
    a = np.random.randn(10,100,60,5).astype(np.float32)

    for i in range(a.ndim):
        assert equal_sum_over_axis(a, axis=i)

def test_sum_all_axes():
    a = np.random.randn(5,10,20,25).astype(np.float32)
    assert equal_sum_over_all_axes(a)

def test_broadcasting_add():
    a = np.array([[ 0.0,  0.0,  0.0],
               [10.0, 10.0, 10.0],
               [20.0, 20.0, 20.0],
               [30.0, 30.0, 30.0]])

    b = np.array([1.0, 2.0, 3.0])
    assert equal_add(a, b)

    a = np.array([0.0, 10.0, 20.0, 30.0])
    a = a[:, np.newaxis]
    b = np.array([1.0, 2.0, 3.0])
    assert equal_add(a, b)

    a = np.random.randn(5,5,5)
    b = np.random.randn(5,5)
    assert equal_add(a, b)

    a = np.random.randn(5,5)
    b = np.random.randn(5,5,5)
    assert equal_add(a, b)

    a = np.random.randn(3, 1, 2)
    b = np.random.randn(3, 1)
    assert equal_add(a, b)

    a = np.random.randn(3, 4)
    b = np.random.randn(2, 3, 4)
    assert equal_add(a, b)
