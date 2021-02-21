import numpy as np
from simplegrad.tensor import Tensor
import torch

def test_simple_conv1d():

    # ===== pytorch =====
    input_1d = np.array([[[1,2,3,4,5]]], dtype='float64')
    param_1d = np.array([[[4,3]]], dtype='float64')

    param_p = torch.tensor(param_1d, requires_grad=True)
    conv1d_p = torch.tensor(input_1d, requires_grad=True)
    a = torch.nn.functional.conv1d(conv1d_p, param_p).sum()
    a.backward()

    # ===== simplegrad =====
    param_s = Tensor(param_1d)
    conv1d_s = Tensor(input_1d)
    conv1d_s.sliding_window(conv1d_s.dot, param_s, subscripts='...i,...i->...', 
                            kernel_size=(2,), stride=1).sum()
    conv1d_s.backward()

    assert np.allclose(a.detach().numpy(), conv1d_s.val)
    assert np.allclose(conv1d_p.grad, conv1d_s.grad)
    assert np.allclose(param_p.grad, param_s.grad)

def test_simple_conv2d():
    input_2d = np.array([

                 [[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 1]],

                 [[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 1]],

                 ], dtype='float')
    
    param_2d = np.array([

                   [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]],

                   [[1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]],

                   ], dtype='float')

    # ===== pytorch =====
    x = input_2d + np.zeros(((1,) + input_2d.shape))
    w = param_2d + np.zeros(((1,) + param_2d.shape))

    x_pt = torch.tensor(x, requires_grad=True)
    w0 = torch.tensor(w, requires_grad=True)
    a = torch.nn.functional.conv2d(x_pt, w0, padding=0)
    a = a.sum()
    a.backward()

    # ===== simplegrad =====
    x = Tensor(x)
    w = Tensor(w)
    x.conv2d(w).sum()
    x.backward()

    # assert a.detach().numpy().ndim == x.val.ndim
    assert (a.detach().numpy() == x.val).all()
    assert (w0.grad.detach().numpy() == w.grad).all()
    assert (x_pt.grad.detach().numpy() == x.grad).all()

def test_conv2d():
    image  = np.random.ranf([10, 16, 20, 40]) # N, in_channels, Hin, Win
    # image = np.random.ranf([20, 16, 50, 100])
    filter = np.random.ranf([20, 16, 3, 3]) # out_channels, in_channels, kernel_size[0], kernel_size[1]

    # ===== pytorch =====
    x = torch.tensor(image, requires_grad=True)
    w0 = torch.tensor(filter, requires_grad=True)
    a = torch.nn.functional.conv2d(x, w0, padding=0).sum()
    a.backward()

    # ===== simplegrad =====
    c = Tensor(filter)
    out_simplegrad = Tensor(image).conv2d(c).sum()
    out_simplegrad.backward()

    assert np.allclose(a.detach().numpy(), out_simplegrad.val)
    assert np.allclose(w0.grad.detach().numpy(), c.grad)
    assert np.allclose(x.grad.detach().numpy(), out_simplegrad.grad)

def test_simple_maxpool1d():
    input_1d = np.array([[[1,2,3,4,5]]], dtype='float64')

    # ===== simplegrad =====
    maxpool1d = Tensor(input_1d)
    maxpool1d.sliding_window(maxpool1d.max, kernel_size=(3,), stride=1).sum()
    maxpool1d.backward()

    # ===== pytorch =====
    input_1d = np.array([[[1,2,3,4,5]]], dtype='float64')
    input_1d = torch.tensor(input_1d, requires_grad=True)
    maxpool1d_p = torch.nn.MaxPool1d(3, stride=1)
    maxpool1d_p = maxpool1d_p(input_1d).sum()
    maxpool1d_p.backward()

    assert np.allclose(input_1d.grad, maxpool1d.grad)

def test_simple_maxpool2d():
        input_2d = np.array([

                 [[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 1]],

                 ], dtype='float')

        # ===== pytorch =====
        _x = input_2d + np.zeros(((1,) + input_2d.shape))
        x = torch.tensor(_x, requires_grad=True)
        maxpool2d_p = torch.nn.MaxPool2d((3,3), stride=1)
        maxpool2d_p = maxpool2d_p(x).sum()
        maxpool2d_p.backward()

        # ===== simplegrad =====
        maxpool2d = Tensor(_x)
        maxpool2d.sliding_window(maxpool2d.max, kernel_size=(3,3), stride=1).sum()
        maxpool2d.backward()

        assert np.allclose(maxpool2d.grad, x.grad)

def test_maxpool2d():
    image  = np.random.ranf([10, 20, 20, 40]) # N, in_channels, Hin, Win

    # ===== pytorch =====
    x = torch.tensor(image, requires_grad=True)
    maxpool2d_p = torch.nn.MaxPool2d((3,3), stride=1)
    maxpool2d_p = maxpool2d_p(x).sum()
    maxpool2d_p.backward()

    # ===== simplegrad =====
    maxpool2d = Tensor(image)
    maxpool2d.sliding_window(maxpool2d.max, kernel_size=(3,3), stride=1).sum()
    maxpool2d.backward()

    assert np.allclose(maxpool2d_p.detach().numpy(), maxpool2d.val)
    assert np.allclose(maxpool2d.grad, x.grad)


def test_simple_conv1d_maxpool1d():

    # ===== pytorch =====
    param_1d = np.array([[[3,4,5]]], dtype='float64')
    input_1d = np.array([[[1,2,3,4,5]]], dtype='float64')

    input_p = torch.tensor(input_1d)
    param_p = torch.tensor(param_1d, requires_grad=True)

    conv1d_p = torch.tensor(input_p, requires_grad=True)
    a = torch.nn.functional.conv1d(conv1d_p, param_p)
    maxpool1d_p = torch.nn.MaxPool1d(3, stride=1)
    maxpool1d_p = maxpool1d_p(a)
    maxpool1d_p.backward()

    # ===== simplegrad =====
    param_s = Tensor(param_1d)
    conv1d_s = Tensor(input_1d)
    conv1d_s.sliding_window(conv1d_s.dot, param_s, subscripts='...i,...i->...', kernel_size=(3,), stride=1)
    maxpool1d_s = conv1d_s.sliding_window(conv1d_s.max, kernel_size=(3,), stride=1)
    maxpool1d_s.backward()

    assert np.allclose(maxpool1d_p.detach().numpy(), maxpool1d_s.val)
    assert np.allclose(conv1d_p.grad, conv1d_s.grad)
    assert np.allclose(param_p.grad, param_s.grad)
