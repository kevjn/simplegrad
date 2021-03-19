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

def equal_einsum(subscripts, a: np.ndarray, b: np.ndarray):
    res_gpu = Tensor(a).dot(Tensor(b), subscripts=subscripts).val
    res_cpu = np.einsum(subscripts, a, b)
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def test_sum_reduction_kernel():
    a = np.random.randn(10,100,60,5).astype(np.float32)

    for i in range(a.ndim):
        assert equal_sum_over_axis(a, axis=i)

def test_sum_all_axes():
    a = np.random.randn(5,10,20,25).astype(np.float32)
    assert equal_sum_over_all_axes(a)

def test_sum_multiple_axes_at_once():
    a = np.random.randn(10,50,30,25).astype(np.float32)

    assert equal_sum_over_axis(a, axis=(0,1))

    assert equal_sum_over_axis(a, axis=(1,0))

    assert equal_sum_over_axis(a, axis=(1,0,2))

    assert equal_sum_over_axis(a, axis=(2,0,1,3))

def test_sum_all_axes_at_once():
    a = np.random.randn(5,25,10,30).astype(np.float32)

    res_cpu = np.sum(a, axis=None)
    res_gpu = Tensor(a).sum(axis=None).val

    assert np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def test_einsum_reduction():
    a = np.random.randn(10,10).astype(np.float32)
    b = np.random.randn(10,10).astype(np.float32)

    # Matrix multiplication
    assert equal_einsum("ij,jk->ik", a, b)

    # Matrix multiplication with transposed output
    assert equal_einsum("ij,jk->ki", a, b)

    # Matrix multiplication with transposed a operand
    assert equal_einsum("ji,jk->ik", a, b)

    # Matrix multiplication with transposed b operand
    assert equal_einsum("ij,kj->ik", a, b)

    # Matrix multiplication with transposed a and b operand
    assert equal_einsum("ji,kj->ik", a, b)

    # Matrix multiplication with transposed output and operand
    assert equal_einsum("ji,kj->ki", a, b)

def test_einsum_with_no_reduction():
    a = np.random.randn(10,10).astype(np.float32)
    b = np.random.randn(10,10).astype(np.float32)

    # Matrix hadamard product
    assert equal_einsum("ij,ij->ij", a, b)

    assert equal_einsum("ji,ij->ji", a, b)

    assert equal_einsum("ij,ji->ji", a, b)

    assert equal_einsum("ji,ji->ji", a, b)

    assert equal_einsum("ji,ji->ij", a, b)

    assert equal_einsum("ij,ij->ji", a, b)

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

def test_simple_backward_with_broadcasting():
    a = np.random.randn(10,10)
    w = np.random.randn(10,10,10)

    tensor_p = (t_p := torch.tensor(a, requires_grad=True)).add(w_p := torch.tensor(w, requires_grad=True)).relu().sum(axis=0).sum(axis=0).sum(axis=0)
    tensor_p.backward()
    tensor = Tensor(a).add(w_s := Tensor(w)).relu().sum(axis=0).sum(axis=0).sum(axis=0)
    tensor.backward()

    assert np.allclose(tensor.val.get(), tensor_p.data.numpy())
    assert np.allclose(tensor.grad.get(), t_p.grad.data.numpy())
    assert np.allclose(w_s.grad.get(), w_p.grad.data.numpy())