from simplegrad import Tensor, Adam

import numpy as np
np.random.seed(100)

def test_tanh():
    a = np.random.randn(10)

    out = Tensor(Tensor.device.Array(a, 'a')).tanh()

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.symbolic))

def test_einsum():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)

    out = Tensor(Tensor.device.Array(a, 'a')).einsum('ij,jk->ik', Tensor(Tensor.device.Array(w0, 'w0')))

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.symbolic))

def test_logsoftmax():
    a = np.random.randn(30, 3)

    out = Tensor(Tensor.device.Array(a, 'a')).logsoftmax().sum()
    out.backward()

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.symbolic))
    assert np.allclose(out.grad.view(np.ndarray), eval(out.grad.symbolic))

def test_maxpool2d():
    image  = np.random.ranf([2, 2, 4, 4]).astype(np.float32)

    out = Tensor(Tensor.device.Array(image, 'image')).maxpool2d().sum()
    out.backward()

    assert (out.data.view(np.ndarray) == eval(out.data.symbolic)).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.symbolic)).all()

def test_conv2d():
    image  = np.random.ranf([10, 16, 20, 40]).astype(np.float32) # N, in_channels, Hin, Win
    filter = np.random.ranf([20, 16, 3, 3]).astype(np.float32) # out_channels, in_channels, kernel_size[0], kernel_size[1]

    w = Tensor(Tensor.device.Array(filter, 'filter'))
    out = Tensor(Tensor.device.Array(image, 'image')).conv2d(w).sum()
    out.backward()

    assert (out.data.view(np.ndarray) == eval(out.data.symbolic)).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.symbolic)).all()

def test_simple_classifier():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)
    b0 = np.random.randn(32).astype(np.float32)
    w1 = np.random.randn(32, 3).astype(np.float32)
    b1 = np.random.randn(3,).astype(np.float32)

    y = np.random.randn(30, 3).astype(np.float32)

    tw0 = Tensor(Tensor.device.Array(w0, 'w0'))
    tw1 = Tensor(Tensor.device.Array(w1, 'w1'))

    tb0 = Tensor(Tensor.device.Array(b0, 'b0'))
    tb1 = Tensor(Tensor.device.Array(b1, 'b1'))

    optim = Adam([tw0, tb0, tw1, tb1])

    for epoch in range(1):
        out = Tensor(Tensor.device.Array(a, 'a')).einsum("ij,jk->ik", tw0)\
            .add(tb0).relu()\
            .einsum("ij,jk->ik", tw1).add(tb1)
        out = out.logsoftmax()

        loss = Tensor(Tensor.device.Array(y, 'y')).mul(out).mul(Tensor(-1.0)).sum(axis=1).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    assert (out.data.view(np.ndarray) == eval(out.data.symbolic)).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.symbolic)).all()
    assert (loss.data.view(np.ndarray) == eval(loss.data.symbolic)).all()
    assert (loss.grad.view(np.ndarray) == eval(loss.grad.symbolic)).all()