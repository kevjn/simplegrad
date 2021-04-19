import numpy as np
from simplegrad import Device, Tensor
import torch

np.random.seed(1337)
Tensor.device = Device.GPU()

def equal_sum_over_axis(arr, *, axis):
    res_cpu = np.sum(arr, axis=axis)
    res_gpu = Tensor(arr).sum(axis=axis).data
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def equal_sum_over_all_axes(arr):
    res_cpu = np.sum(arr)
    a = Tensor(arr)
    for _ in range(arr.ndim):
        res_gpu = a.sum(axis=0)
    res_gpu = res_gpu.data
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def equal_add(a: np.ndarray, b: np.ndarray):
    tensor_pytorch = torch.tensor(a).add(torch.tensor(b))
    tensor_simplegrad = Tensor(a).add(Tensor(b))
    return np.allclose(tensor_simplegrad.data.get(), tensor_pytorch.data.numpy())

def equal_einsum(subscripts, a: np.ndarray, b: np.ndarray):
    res_gpu = Tensor(a).einsum(subscripts, Tensor(b)).data
    res_cpu = np.einsum(subscripts, a, b)
    return np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-06)

def test_sum_reduction_kernel():
    a = np.random.randn(10,100,60,5).astype(np.float32)

    for i in range(a.ndim):
        assert equal_sum_over_axis(a, axis=i)

def test_sum_all_axes():
    a = np.random.randn(10,100,200,255).astype(np.float32)

    res_gpu = Tensor(a).sum(axis=(0,1,2,3)).data
    res_cpu = np.sum(a, axis=None)
    assert np.allclose(res_cpu, res_gpu.get(), rtol=1e-04, atol=1e-07)

def test_sum_multiple_axes_at_once():
    a = np.random.randn(10,50,30,25).astype(np.float32)

    assert equal_sum_over_axis(a, axis=(0,1))

    assert equal_sum_over_axis(a, axis=(1,0))

    assert equal_sum_over_axis(a, axis=(1,0,2))

    assert equal_sum_over_axis(a, axis=(2,0,1,3))

def test_sum_all_axes_at_once():
    a = np.random.randn(5,25,10,30).astype(np.float32)

    res_cpu = np.sum(a, axis=None)
    res_gpu = Tensor(a).sum(axis=None).data

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

    a = np.random.randn(10,10,10).astype(np.float32)
    b = np.random.randn(10,10,10).astype(np.float32)

    # Batch matrix multiplication
    assert equal_einsum("bij,bjk->bik", a, b)

def test_einsum_reduction_with_unequal_ndim():
    a = np.random.randn(10,10,10).astype(np.float32)
    b = np.random.randn(10,10).astype(np.float32)

    assert equal_einsum("ijk,jk->ik", a, b)

    assert equal_einsum("ijk,ji->ik", a, b)

    assert equal_einsum("ijk,ij->ik", a, b)

    assert equal_einsum("ij,kij->ik", b, a)

    assert equal_einsum("kij,ij->ik", a, b)

    assert equal_einsum("kji,ij->ik", a, b)

    assert equal_einsum("kji,ji->ik", a, b)

    assert equal_einsum("jik,ji->ik", a, b)

    k = np.random.randn(100, 100, 2)
    w = np.random.randn(2, 64)
    assert equal_einsum("ijk,kl->ijl", k, w)

    # assert equal_einsum("jki,ji->ik", a, b) # TODO: fix this
    # assert equal_einsum("ijk,jl->ik", a, b) # TODO: fix this

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

    a = np.random.randn(1)
    b = np.random.randn(3,3,3)
    assert equal_add(a, b)

def test_simple_backward_with_broadcasting():
    a = np.random.randn(10,10)
    w = np.random.randn(10,10,10)

    tensor_p = (t_p := torch.tensor(a, requires_grad=True)).add(w_p := torch.tensor(w, requires_grad=True)).relu().sum(axis=0).sum(axis=0).sum(axis=0)
    tensor_p.backward()
    tensor = Tensor(a).add(w_s := Tensor(w)).relu().sum(axis=0).sum(axis=0).sum(axis=0)
    tensor.backward()

    assert np.allclose(tensor.data.get(), tensor_p.data.numpy())
    assert np.allclose(tensor.grad.get(), t_p.grad.data.numpy())
    assert np.allclose(w_s.grad.get(), w_p.grad.data.numpy())

def test_pow_backward_pass():
    a = np.random.randn(10,10)

    tp = torch.tensor(a, requires_grad=True)
    tp.pow(torch.tensor(2)).sum().backward()

    ts = Tensor(a).pow(Tensor(2))
    ts.backward()

    np.allclose(tp.grad.data.numpy(), ts.grad.get())

def test_forward_and_backward_for_simple_classifier():
    X = np.random.randn(3000, 2)
    w0 = np.random.randn(2,64)
    b0 = np.random.randn(64)
    w1 = np.random.randn(64, 3)
    b1 = np.random.randn(3)
    y = np.random.randn(3000, 3)

    # ========== pytorch =========
    tp = torch.tensor(X, requires_grad=True)
    outp = tp.matmul(w0p := torch.tensor(w0, requires_grad=True)).add(torch.tensor(b0))\
        .relu().matmul(w1p := torch.tensor(w1, requires_grad=True)).add(torch.tensor(b1))
    
    outp = torch.functional.F.log_softmax(outp)

    # Negative LL loss
    lossp = torch.tensor(y).mul(outp).mul(torch.tensor(-1.0)).sum(axis=1).mean()
    lossp.backward() 

    # ========== simplegrad ==========
    outs = Tensor(X).einsum("ij,jk->ik", w0s := Tensor(w0))\
        .add(Tensor(b0)).relu().einsum("ij,jk->ik", w1s := Tensor(w1))\
        .add(Tensor(b1))

    outs.logsoftmax()
    
    # Negative LL loss
    loss = Tensor(y).mul(outs).mul(Tensor(-1)).sum(axis=1).mean()
    loss.backward()

    assert np.allclose(outs.data.get(), outp.data.numpy(), rtol=1e-04, atol=1e-05)
    assert np.allclose(outs.grad.get(), tp.grad.data.numpy(), rtol=1e-04, atol=1e-05)
    assert np.allclose(w0s.grad.get(), w0p.grad.data.numpy(), rtol=1e-04, atol=1e-05)
    assert np.allclose(w1s.grad.get(), w1p.grad.data.numpy(), rtol=1e-04, atol=1e-05)

def test_train_simple_classifier():
    from test.test_tensor import test_train_simple_classifier
    test_train_simple_classifier()
    assert True
