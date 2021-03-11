from simplegrad.tensor import Device, Tensor
from simplegrad import optim as optimizer

import numpy as np
np.random.seed(42)

import torch
from torch import nn
import matplotlib.pyplot as plt
import pytest

def scaled_dot_product_pytorch(q, k, v, mask=None, grad=True):
    q = torch.tensor(q, requires_grad=True)
    k = torch.tensor(k, requires_grad=True)
    v = torch.tensor(v, requires_grad=True)

    d_k = q.size()[-1]
    attn_logits = torch.einsum("ijkl,ijml->ijkm", [q,k])
    attn_logits = attn_logits / np.sqrt(d_k)

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = torch.functional.F.softmax(attn_logits, dim=-1)

    values = torch.matmul(attention, v)

    if grad:
        # Can only call backward on scalar values
        values = values.sum()
        values.backward()
        return values, attention, q.grad, k.grad, v.grad
    return values, attention

def scaled_dot_product_simplegrad(q, k, v, mask=None, grad=True):
    
    d_k = q.shape[-1]
    attn_logits = q.dot(k, subscripts="ijkl,ijml->ijkm")
    attn_logits = attn_logits.div(Tensor(np.sqrt(d_k)))

    if mask is not None:
        raise NotImplementedError

    attention = attn_logits.softmax()
    values = attention.fork().dot(v, subscripts="ijkl,ijlm->ijkm")

    if grad:
        # To test backward pass we need to sum here
        values = values.sum()
        values.backward()
        attention.backward() # since value is forked we need to explicitly call backward here
        return values, attention, q.grad, k.grad, v.grad
    return values, attention

def multi_head_attention_forward_simplegrad(q, k, v, embed_dim, num_heads, *proj_weights, out_proj_weight):
        q, k, v = *(x.fork().dot(w, subscripts="ijk,lmk->jlim") for x,w in zip([q,k,v], proj_weights)),

        output, attn = scaled_dot_product_simplegrad(q, k, v, grad=False)

        return output.dot(out_proj_weight, subscripts="ijkl,mjl->kim"), attn

def test_scaled_dot_product_attention():
    def compare(pytorch_tensors, simplegrad_tensors):
        t1, t2, *grad1 = pytorch_tensors
        tensor1 = (t1, t2)
        t1,t2, *grad2 = simplegrad_tensors
        tensor2 = (t1, t2)

        equal_out = all(np.allclose(t1.data.numpy(), t2.val) for t1, t2 in zip(tensor1, tensor2))
        equal_grad = all(np.allclose(t1.data.numpy(), t2) for t1, t2 in zip(grad1, grad2))
        return equal_out and equal_grad

    seq_len, d_k = 3, 2 # data hyperparameters

    q = np.random.randn(1, 1, seq_len, d_k)
    k = np.random.randn(1, 1, seq_len, d_k)
    v = np.random.randn(1, 1, seq_len, d_k)

    assert compare(scaled_dot_product_pytorch(q, k, v),
                   scaled_dot_product_simplegrad(Tensor(q), Tensor(k), Tensor(v)))


def test_multi_head_self_attention():

    def forward_pytorch(q, k, v, embed_dim, num_heads, q_proj_weight, 
                        k_proj_weight, v_proj_weight, out_proj_weight):
        batch_size, seq_len = q.size(0), q.size(1)
        head_dim = embed_dim // num_heads

        q = torch.einsum("ijk,lmk->jlim", [q, q_proj_weight])
        k = torch.einsum("ijk,lmk->jlim", [k, k_proj_weight])
        v = torch.einsum("ijk,lmk->jlim", [v, v_proj_weight])

        output, attn = scaled_dot_product_pytorch(q, k, v, grad=False)

        return torch.einsum("ijkl,mjl->kim", [output, out_proj_weight]), attn

    batch_size, seq_len, num_categories = 1, 16, 10 # data hyperparams

    # One input example
    X = torch.randint(num_categories, size=(batch_size, seq_len))
    X = torch.functional.F.one_hot(X, num_classes=num_categories).float() # [BatchSize, seq_len, embed_dim]

    num_heads, embed_dim = 2, 10 # Model hyperparams
    head_dim = embed_dim // num_heads

    # reshape input to comply with pytorch
    X = X.reshape(seq_len, batch_size, embed_dim)

    # weight params
    k = 1 / embed_dim
    q_proj_weight = torch.FloatTensor(embed_dim, embed_dim).uniform_(-np.sqrt(k), np.sqrt(k))
    k_proj_weight = torch.FloatTensor(embed_dim, embed_dim).uniform_(-np.sqrt(k), np.sqrt(k))
    v_proj_weight = torch.FloatTensor(embed_dim, embed_dim).uniform_(-np.sqrt(k), np.sqrt(k))

    out_proj_weight = torch.FloatTensor(embed_dim, embed_dim).uniform_(-np.sqrt(k), np.sqrt(k))

    # requires grad for backprop
    for weight in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight):
        weight.requires_grad = True

    # Compare the two pytorch models
    out_pytorch1, _ = torch.functional.F.multi_head_attention_forward(
                    X, X, X, embed_dim, num_heads, None, None, None, None, False,
                    0.0, out_proj_weight, None, q_proj_weight = q_proj_weight, 
                    k_proj_weight = k_proj_weight, v_proj_weight = v_proj_weight,
                    use_separate_proj_weight=True)

    out_pytorch1.sum().backward()
    pytorch_grad = [w.grad.clone().data.numpy() for w in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)]

    # reshape weights here to make better use of einsum (decompose embed_dim)
    q_proj_weight = q_proj_weight.reshape(num_heads, head_dim, embed_dim)
    k_proj_weight = k_proj_weight.reshape(num_heads, head_dim, embed_dim)
    v_proj_weight = v_proj_weight.reshape(num_heads, head_dim, embed_dim)

    out_proj_weight = out_proj_weight.reshape(embed_dim, num_heads, head_dim)

    out_pytorch2, _ = forward_pytorch(X, X, X, embed_dim, num_heads, q_proj_weight,
                        k_proj_weight, v_proj_weight, out_proj_weight)

    assert torch.allclose(out_pytorch1, out_pytorch2)

    # compare with simeplgrad
    # convert to simplegrad tensors
    q_proj_weight = Tensor(q_proj_weight.data.numpy())
    k_proj_weight = Tensor(k_proj_weight.data.numpy())
    v_proj_weight = Tensor(v_proj_weight.data.numpy())

    out_proj_weight = Tensor(out_proj_weight.data.numpy())

    X = Tensor(X.data.numpy())

    out_simplegrad, attn = multi_head_attention_forward_simplegrad(X, X, X, embed_dim, num_heads,
                            q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight=out_proj_weight)

    assert np.allclose(out_pytorch1.data.numpy(), out_simplegrad.val)

    # test backward pass
    out_simplegrad.sum().backward()
    attn.backward()

    grad_in_proj_simplegrad = *(w.grad.reshape(embed_dim, embed_dim) for w in [q_proj_weight, k_proj_weight, v_proj_weight]),
    grad_out_proj_simplegrad = out_proj_weight.grad.reshape(embed_dim, embed_dim),

    grad_simplegrad = grad_in_proj_simplegrad + grad_out_proj_simplegrad

    assert np.allclose(pytorch_grad, grad_simplegrad, rtol=1e-04, atol=1e-07)

def test_layer_norm():
    input = torch.randn(20, 5, 10, 10)
    layer_norm = nn.LayerNorm([10], elementwise_affine=False)
    output_pytorch = layer_norm(input)
    out_simplegrad = Tensor(input.data.numpy()).layer_norm(-1, Tensor(1), Tensor(0))

    assert np.allclose(output_pytorch.data.numpy(), out_simplegrad.val, rtol=1e-04, atol=1e-07)

def test_layer_norm_with_grad():
    input = torch.randn(20, 5, 10, 10)
    layer_norm = nn.LayerNorm([10, 10], elementwise_affine=True)
    layer_norm.weight = torch.nn.Parameter(torch.randn(layer_norm.weight.shape))
    output_pytorch = layer_norm(input)

    simplegrad_weight = Tensor(layer_norm.weight.data.numpy())
    out_simplegrad = Tensor(input.data.numpy()).layer_norm((-1,-2), simplegrad_weight, Tensor(0))

    assert np.allclose(output_pytorch.data.numpy(), out_simplegrad.val, rtol=1e-04, atol=1e-07)

    output_pytorch.sum().backward()
    out_simplegrad.sum().backward()
    assert np.allclose(layer_norm.weight.grad.data.numpy(), simplegrad_weight.grad, rtol=1e-04, atol=1e-07)
