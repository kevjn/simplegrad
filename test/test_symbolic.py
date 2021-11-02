from simplegrad import Tensor, Adam
from accel.symbolic import SymbolicArray

import numpy, numpy as np
np.random.seed(100)

def test_tanh():
    a = np.random.randn(10)

    out = Tensor(SymbolicArray(a, 'a')).tanh()

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.symbolic))

def test_einsum():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)

    out = Tensor(SymbolicArray(a, 'a')).einsum('ij,jk->ik', Tensor(SymbolicArray(w0, 'w0')))

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.symbolic))

def test_logsoftmax():
    a = np.random.randn(30, 3)

    out = Tensor(SymbolicArray(a, 'a')).logsoftmax().sum()
    out.backward()

    assert np.allclose(out.data.view(np.ndarray), eval(out.data.expand()))
    assert np.allclose(out.grad.view(np.ndarray), eval(out.grad.expand()))

def test_maxpool2d():
    image  = np.random.ranf([2, 2, 4, 4]).astype(np.float32)

    out = Tensor(SymbolicArray(image, 'image')).maxpool2d().sum()
    out.backward()

    assert (out.data.view(np.ndarray) == eval(out.data.expand())).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.expand())).all()

def test_conv2d():
    image  = np.random.ranf([10, 16, 20, 40]).astype(np.float32) # N, in_channels, Hin, Win
    filter = np.random.ranf([20, 16, 3, 3]).astype(np.float32) # out_channels, in_channels, kernel_size[0], kernel_size[1]

    w = Tensor(SymbolicArray(filter, 'filter'))
    out = Tensor(SymbolicArray(image, 'image')).conv2d(w).sum()
    out.backward()

    assert (out.data.view(np.ndarray) == eval(out.data.expand())).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.expand())).all()

def test_simple_classifier():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)
    b0 = np.random.randn(32).astype(np.float32)
    w1 = np.random.randn(32, 3).astype(np.float32)
    b1 = np.random.randn(3,).astype(np.float32)

    y = np.random.randn(30, 3).astype(np.float32)

    tw0 = Tensor(SymbolicArray(w0, 'w0'))
    tw1 = Tensor(SymbolicArray(w1, 'w1'))

    tb0 = Tensor(SymbolicArray(b0, 'b0'))
    tb1 = Tensor(SymbolicArray(b1, 'b1'))

    optim = Adam([tw0, tb0, tw1, tb1])

    out = Tensor(SymbolicArray(a, 'a')).einsum("ij,jk->ik", tw0)\
        .add(tb0).relu()\
        .einsum("ij,jk->ik", tw1).add(tb1)
    out = out.logsoftmax()

    loss = Tensor(SymbolicArray(y, 'y')).mul(out).mul(-1.0).sum(axis=1).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    prg = "\n".join(f"{k} = {v}" for k,v in out.grad.locals.items())
    exec(prg)
    assert (out.grad.view(np.ndarray) == eval(out.grad.symbolic)).all()

    assert (out.data.view(np.ndarray) == eval(out.data.expand())).all()
    assert (out.grad.view(np.ndarray) == eval(out.grad.expand())).all()
    assert (loss.data.view(np.ndarray) == eval(loss.data.expand())).all()
    assert (loss.grad.view(np.ndarray) == eval(loss.grad.expand())).all()