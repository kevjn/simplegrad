from simplegrad import Device, Tensor, Adam

import numpy as np
np.random.seed(100)

import torch
import matplotlib.pyplot as plt
import pytest

from nnfs.datasets import spiral_data

import cv2

import tensorflow as tf

def test_sigmoid():
    a = np.random.randn(10)
    b = torch.tensor(a, requires_grad=True)
    c = torch.sigmoid(b).sum()
    c.backward()

    d = Tensor(a).sigmoid().sum()
    d.backward()

    assert np.allclose(b.grad, d.grad)

def test_tanh():
    a = np.random.randn(10)
    b = torch.tensor(a, requires_grad=True)
    c = torch.tanh(b).sum()
    c.backward()

    d = Tensor(a)
    e = d.tanh().sum()
    e.backward()

    assert np.allclose(c.detach().numpy(), e.data)
    assert np.allclose(b.grad, d.grad)

def test_multiple_tanh():
    a = np.random.randn(10)
    b = torch.tensor(a, requires_grad=True)
    c = b.tanh().tanh().tanh().tanh().sum()
    c.backward()

    d = Tensor(a)
    e = d.tanh().tanh().tanh().tanh().sum()
    e.backward()

    assert np.allclose(c.detach().numpy(), e.data)
    assert np.allclose(b.grad, d.grad)

def test_tanh_positive_overflow():
    a = np.array([10, 20, 80, 90, 200, 200, 600, 800], dtype=float)
    b = torch.tensor(a, requires_grad=True)
    c = b.tanh()

    d = Tensor(a, 'a')
    e = d.tanh()

    assert np.allclose(c.data.numpy(), e.data.view(np.ndarray))

    c.sum().backward()
    e.sum().backward()

    assert np.allclose(b.grad.numpy(), d.grad.view(np.ndarray))

def test_tanh_negative_overflow():
    a = np.array([-10, -20, -80, -90, -200, -200, -600, -800], dtype=float)
    b = torch.tensor(a, requires_grad=True)
    c = b.tanh()

    d = Tensor(a, 'a')
    e = d.tanh()

    assert np.allclose(c.data.numpy(), e.data.view(np.ndarray))

    c.sum().backward()
    e.sum().backward()

    assert np.allclose(b.grad.numpy(), d.grad.view(np.ndarray))

def test_softmax_and_mean():
    from scipy.special import softmax

    X, y = spiral_data(samples=10, classes=3)

    # one-hot encode sparse y
    y = Tensor(np.eye(len(np.unique(y)))[y])

    w0 = Tensor(np.random.randn(2,32))
    w1 = Tensor(np.random.randn(32,3))

    before_softmax = Tensor(X).einsum('ij,jk->ik', w0).relu().einsum('ij,jk->ik', w1).data
    mdl = Tensor(X).einsum('ij,jk->ik', w0).relu().einsum('ij,jk->ik', w1).softmax()
    out = mdl.data

    assert np.allclose(out, softmax(before_softmax, axis=-1))

    loss = mdl.mul(y).mean()
    assert np.allclose(loss.data, np.mean(out * y.data, axis=-1, keepdims=True))

def test_simple_neuron_backward():
    X = np.array([1, -2, 3, 1])
    w = Tensor(np.array([-3, -1, 2, 1]))

    tensor = Tensor(X).mul(w).sum().relu()
    tensor.backward()

    assert (w.grad == np.array([1, -2, 3, 1])).all()
    assert (tensor.grad == np.array([-3, -1, 2, 1])).all()


def test_simple_backward():

    x = torch.eye(3, requires_grad=True)
    y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()

    x2 = Tensor(np.eye(3))
    y2 = Tensor(np.array([[2.0, 0, -2.0]]))
    z = y2.einsum("ij,jk->ik", x2).sum()
    z.backward()

    assert np.allclose(x.grad, x2.grad)
    assert np.allclose(y.grad, y2.grad)

def test_backward_max():
    x = torch.eye(3, requires_grad=True)
    y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x).max()

    z.backward()

    x2 = Tensor(np.eye(3))
    y2 = Tensor(np.array([[2.0, 0, -2.0]]))
    z = y2.einsum("ij,jk->ik", x2).max()
    z.backward()

    assert np.allclose(x.grad, x2.grad)
    assert np.allclose(y.grad, y2.grad)

def test_backward_pass():
    X_init, y = spiral_data(samples=10, classes=3)
    # one-hot encode sparse y
    y_init = np.eye(len(np.unique(y)))[y]

    w0_init = np.random.randn(2,32)
    w1_init = np.random.randn(32,3)

    def test_simplegrad():
        y = Tensor(y_init)
        w0 = Tensor(w0_init)
        w1 = Tensor(w1_init)

        out = Tensor(X_init).einsum("ij,jk->ik", w0)
        out = out.logsoftmax()
        out = out.sum()
        out.backward()
        return w0.grad, out.grad

    def test_pytorch():
        y = torch.tensor(y_init)
        x = torch.tensor(X_init, requires_grad=True)
        w0 = torch.tensor(w0_init, requires_grad=True)
        w1 = torch.tensor(w1_init, requires_grad=True)

        out = x.matmul(w0)
        out = torch.nn.functional.log_softmax(out, dim=1)
        out = out.sum()
        out.backward()
        return w0.grad, x.grad
        return out.detach().numpy(), w0.grad

    for a,b in zip(test_pytorch(), test_simplegrad()):
        assert np.allclose(a,b)

def test_backward_pass_with_loss():
    X_init, y = spiral_data(samples=10, classes=3)
    # one-hot encode sparse y
    y_init = np.eye(len(np.unique(y)))[y]

    w0_init = np.random.randn(2,32)
    w1_init = np.random.randn(32,3)

    def test_simplegrad():
        y = Tensor(y_init)
        w0 = Tensor(w0_init)
        w1 = Tensor(w1_init)

        out = Tensor(X_init).einsum("ij,jk->ik", w0).relu().einsum("ij,jk->ik", w1).logsoftmax()
        loss = out.mul(y).mul(Tensor(-1.)).sum(axis=1).mean()
        loss.backward()
        return w0.grad, w1.grad

    def test_pytorch():
        y = torch.tensor(y_init)
        x = torch.tensor(X_init, requires_grad=True)
        w0 = torch.tensor(w0_init, requires_grad=True)
        w1 = torch.tensor(w1_init, requires_grad=True)

        out = x.matmul(w0).relu().matmul(w1)
        out = torch.nn.functional.log_softmax(out, dim=1)
        loss = out.mul(y).mul(torch.tensor(-1.)).sum(axis=1).mean()
        loss.backward()
        return w0.grad, w1.grad

    for a,b in zip(test_pytorch(), test_simplegrad()):
        assert np.allclose(a,b)

def test_image_conv2d():
    image = cv2.imread('city.jpeg') 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 

    image = image.reshape(-1, 1, *image.shape)

    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    k = k.reshape(-1, 1, *k.shape)

    # Edge Detection Kernel
    kernel = Tensor(k)

    res = Tensor(image).conv2d(kernel, padding=2)

    cv2.imwrite('2DConvolved_test.jpg', res.data.squeeze())

@pytest.mark.slow
def test_train_simple_classifier():
    num_classes = 3

    X, y_true = spiral_data(samples=1000, classes=num_classes)

    # one-hot encode sparse y
    y = np.eye(num_classes)[y_true]

    w0 = Tensor(np.random.randn(2,1024))
    b0 = Tensor(np.random.randn(1024))
    w1 = Tensor(np.random.randn(1024, num_classes))
    b1 = Tensor(np.random.randn(num_classes))

    optim = Adam([w0, b0, w1, b1])

    for epoch in range(200):
        out = Tensor(X).einsum("ij,jk->ik", w0).add(b0).relu().einsum("ij,jk->ik", w1).add(b1)
        out = out.logsoftmax()

        y_pred = out.cpu().argmax(axis=1)
        acc = np.mean(y_pred == y_true)

        # Categorical cross-entropy loss
        loss = Tensor(y).mul(out).mul(Tensor(-1.0)).sum(axis=1).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # if epoch % 500 == 0:
        #     print(f"loss: {loss.data}, acc: {acc}")

    assert loss.data[0] < 0.4 and acc > 0.8

    # visualize decision boundary
    num_points = 100
    x_1 = np.linspace(-1.5, 1.5, num_points)
    x_2 = np.linspace(-1.5, 1.5, num_points)

    X1, Y2 = np.meshgrid(x_1, x_2)
    k = np.dstack((X1, Y2))
    out = Tensor(k).einsum('ijk,kl->ijl', w0).add(b0).relu()\
        .einsum('ijk,kl->ijl', w1).add(b1).softmax()
    res = out.cpu().argmax(axis=-1)

    cs = plt.contourf(X1, Y2, res, cmap="brg", alpha=0.5)
    plt.colorbar(cs)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=5, cmap="brg", alpha=0.5)
    plt.grid()
    plt.show()
