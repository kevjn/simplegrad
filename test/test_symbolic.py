from simplegrad import Device, Tensor

import numpy as np
np.random.seed(100)

def test_tanh():
    a = np.random.randn(10)

    out = Tensor(a, 'a').tanh()

    assert np.allclose(out.data, eval(out.data.symbolic))

def test_einsum():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)

    out = Tensor(a, 'a').einsum('ij,jk->ik', Tensor(w0, 'w0'))

    assert np.allclose(out.data, eval(out.data.symbolic))

def test_logsoftmax():
    a = np.random.randn(30, 3)

    out = Tensor(a, 'a').logsoftmax()

    assert np.allclose(out.data, eval(out.data.symbolic))

def test_simple_classifier():
    a = np.random.randn(30, 2).astype(np.float32)
    w0 = np.random.randn(2, 32).astype(np.float32)
    b0 = np.random.randn(32).astype(np.float32)
    w1 = np.random.randn(32, 3).astype(np.float32)
    b1 = np.random.randn(3,).astype(np.float32)

    y = np.random.randn(30, 3).astype(np.float32)

    out = Tensor(a, 'a').einsum("ij,jk->ik", Tensor(w0, 'w0'))\
        .add(Tensor(b0, 'b0')).relu()\
        .einsum("ij,jk->ik", Tensor(w1, 'w1')).add(Tensor(b1, 'b1'))
    out = out.logsoftmax()

    loss = Tensor(y, 'y').mul(out).mul(Tensor(-1.0)).sum(axis=1).mean()

    # loss.backward()

    assert (out.data == eval(out.data.symbolic)).all()
    assert (loss.data == eval(loss.data.symbolic)).all()