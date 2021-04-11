from simplegrad import Device, Tensor, Adam
import itertools as it

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
    attn_logits = attn_logits.div(Tensor(d_k).pow(Tensor(0.5)))

    if mask is not None:
        attn_logits.add(mask)

    attention = attn_logits.softmax()
    values = attention.fork().dot(v, subscripts="ijkl,ijlm->ijkm")

    if grad:
        # To test backward pass we need to sum here
        values = values.sum()
        values.backward()
        attention.backward() # since value is forked we need to explicitly call backward here
        return values, attention, q.grad, k.grad, v.grad
    return values, attention

def multi_head_attention_forward_simplegrad(q, k, v, in_proj_weights, out_proj_weight, mask=None):
    q, k, v = (x.fork().dot(w, subscripts="ijk,lmk->jlim") for x,w in zip([q,k,v], in_proj_weights))

    output, attn = scaled_dot_product_simplegrad(q, k, v, grad=False, mask=mask)

    return output.dot(out_proj_weight, subscripts="ijkl,mjl->kim"), attn

# skip bias in multi-head attention for now
def encoder_layer_forward_simplegrad(src, *, in_proj_weights, out_proj_weight, linear_weights, layernorm_weights):
    layernorm1_weight, layernorm1_bias, layernorm2_weight, layernorm2_bias = layernorm_weights

    fork1 = src.fork()
    # Multi-Head Attention forward
    output, attn = multi_head_attention_forward_simplegrad(src, src, src, in_proj_weights, out_proj_weight)

    # Add and Norm forward
    src = output.add(fork1)
    src.layer_norm(-1, layernorm1_weight, layernorm1_bias)

    fork2 = src.fork()
    # Feed forward
    w0, b0, w1, b1 = linear_weights
    src.dot(w0, subscripts="ijk,lk->ijl").add(b0).relu()\
       .dot(w1, subscripts="ijk,lk->ijl").add(b1)

    # Add and Norm forward
    src.add(fork2)
    src.layer_norm(-1, layernorm2_weight, layernorm2_bias)

    return src, attn, fork1, fork2

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

    out_simplegrad, attn = multi_head_attention_forward_simplegrad(X, X, X,
                            (q_proj_weight, k_proj_weight, v_proj_weight), out_proj_weight)

    assert np.allclose(out_pytorch1.data.numpy(), out_simplegrad.val)

    # test backward pass
    out_simplegrad.sum().backward()
    attn.backward()

    grad_in_proj_simplegrad = *(w.grad.reshape(embed_dim, embed_dim) for w in [q_proj_weight, k_proj_weight, v_proj_weight]),
    grad_out_proj_simplegrad = out_proj_weight.grad.reshape(embed_dim, embed_dim),

    simplegrad_grad = grad_in_proj_simplegrad + grad_out_proj_simplegrad

    assert np.allclose(pytorch_grad, simplegrad_grad, rtol=1e-07, atol=1e-06)

def test_multi_head_self_attention_with_mask():
    embed_dim = 10
    num_heads = 2
    head_dim = embed_dim // num_heads

    X = torch.randn(3, 1, 10, requires_grad=True) # source sequence length 3, batch size 1, embedding size 10

    attn_mask = (torch.triu(torch.ones(3, 3)) == 1).transpose(0, 1)
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

    torch.manual_seed(42)
    q_proj_weight = torch.randn(10,10, requires_grad=True)
    torch.manual_seed(42)
    k_proj_weight = torch.randn(10,10, requires_grad=True)
    torch.manual_seed(42)
    v_proj_weight = torch.randn(10,10, requires_grad=True)

    torch.manual_seed(42)
    out_proj_weight = torch.randn(10,10, requires_grad=True)

    pytorch_out, _ = torch.functional.F.multi_head_attention_forward(
                    X, X, X, embed_dim, num_heads, None, None, None, None, False,
                    0.0, out_proj_weight, None, q_proj_weight = q_proj_weight, 
                    k_proj_weight = k_proj_weight, v_proj_weight = v_proj_weight,
                    use_separate_proj_weight=True, attn_mask=attn_mask)

    # compute gradients
    pytorch_out.sum().backward()
    pytorch_grad = [w.grad.data.clone().detach().numpy() for w in (X, q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)]

    # Reshape in_proj_weights
    q_proj_weight = q_proj_weight.reshape(num_heads, head_dim, embed_dim)
    k_proj_weight = k_proj_weight.reshape(num_heads, head_dim, embed_dim)
    v_proj_weight = v_proj_weight.reshape(num_heads, head_dim, embed_dim)

    # Reshape out_proj_weight
    out_proj_weight = out_proj_weight.reshape(embed_dim, num_heads, head_dim)

    # convert all params to simplegrad Tensors
    q_proj_weight = Tensor(q_proj_weight.data.clone().numpy())
    k_proj_weight = Tensor(k_proj_weight.data.clone().numpy())
    v_proj_weight = Tensor(v_proj_weight.data.clone().numpy())

    out_proj_weight = Tensor(out_proj_weight.data.clone().numpy())

    X = Tensor(X.data.clone().numpy())

    # gather in_proj_weights
    in_proj_weights = (q_proj_weight, k_proj_weight, v_proj_weight)

    simplegrad_out, attn = multi_head_attention_forward_simplegrad(X, X, X, in_proj_weights, out_proj_weight, mask=Tensor(attn_mask.data.clone().detach()))

    # compute gradients
    simplegrad_out.backward()

    attn._backward(np.zeros(attn.shape))
    X._backward(np.zeros(X.shape))
    simplegrad_grad = [w.grad.reshape(embed_dim, embed_dim) for w in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)]
    simplegrad_grad = simplegrad_grad

    # compare output
    assert np.allclose(simplegrad_out.val, pytorch_out.data.numpy(), atol=1e-7)

    assert np.allclose(X.grad, pytorch_grad[0], atol=1e-5)
    
    # compare gradients
    assert np.allclose(simplegrad_grad, pytorch_grad[1:], atol=1e-4)


def test_layer_norm():
    input = torch.randn(20, 5, 10, 10)
    layer_norm = nn.LayerNorm([10], elementwise_affine=False)
    output_pytorch = layer_norm(input)
    out_simplegrad = Tensor(input.data.numpy()).layer_norm(-1, Tensor(1), Tensor(0))

    assert np.allclose(output_pytorch.data.numpy(), out_simplegrad.val, rtol=1e-06, atol=1e-06)

def test_layer_norm_with_grad():
    input = torch.randn(20, 5, 10, 10)
    layer_norm = nn.LayerNorm([10, 10], elementwise_affine=True)
    layer_norm.weight = torch.nn.Parameter(torch.randn(layer_norm.weight.shape))
    output_pytorch = layer_norm(input)

    simplegrad_weight = Tensor(layer_norm.weight.clone().data.numpy())
    simplegrad_bias = Tensor(layer_norm.bias.clone().data.numpy())
    out_simplegrad = Tensor(input.data.numpy()).layer_norm((-1,-2), simplegrad_weight, simplegrad_bias)

    assert np.allclose(output_pytorch.data.numpy(), out_simplegrad.val, rtol=1e-06, atol=1e-06)

    output_pytorch.sum().backward()
    out_simplegrad.sum().backward()
    assert np.allclose(layer_norm.weight.grad.data.numpy(), simplegrad_weight.grad, rtol=1e-06, atol=1e-05)
    assert np.allclose(layer_norm.bias.grad.data.numpy(), simplegrad_bias.grad, rtol=1e-06, atol=1e-05)

def test_encoder_layer():
    # Model hyperparameters
    embed_dim = 512
    num_heads = 8
    head_dim = embed_dim // num_heads

    encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.0, dim_feedforward=64)
    src = torch.rand(10, 32, 512)
    out_pytorch = encoder_layer(src)

    weights = encoder_layer.parameters()

    # extract Multi-Head Self Attention weights
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, *other = weights

    # extract Linear weights
    linear1_weight, linear1_bias, linear2_weight, linear2_bias, *other = other

    # extract LayerNorm weights
    layernorm1_weight, layernorm1_bias, layernorm2_weight, layernorm2_bias = other

    # get gradients from pytorch
    out_pytorch.sum().backward()
    pytorch_grad = [w.grad.clone().data.numpy() for w in encoder_layer.parameters()]

    # preprocess weights for simplegrad
    # split in_proj_weights into q,k,v
    q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3, dim=0)

    # reshape the first embed_dim dimension into (num_heads, head_dim)
    q_proj_weight = q_proj_weight.reshape(num_heads, head_dim, embed_dim)
    k_proj_weight = k_proj_weight.reshape(num_heads, head_dim, embed_dim)
    v_proj_weight = v_proj_weight.reshape(num_heads, head_dim, embed_dim)

    # also reshape out_proj_weight
    out_proj_weight = out_proj_weight.reshape(embed_dim, num_heads, head_dim)

    # convert all weights to simplegrad Tensors
    *in_proj_weights, out_proj_weight = [Tensor(x.data.numpy()) for x in (q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight)]
    linear_weights = [Tensor(x.data.numpy()) for x in (linear1_weight, linear1_bias, linear2_weight, linear2_bias)]
    layernorm_weights = [Tensor(x.data.numpy()) for x in (layernorm1_weight, layernorm1_bias, layernorm2_weight, layernorm2_bias)]

    # skip bias in multi-head attention for now
    out_simplegrad, attn, fork1, fork2 = encoder_layer_forward_simplegrad(Tensor(src.data.numpy()),
                        in_proj_weights = in_proj_weights,
                        out_proj_weight = out_proj_weight,
                        linear_weights =  linear_weights,
                        layernorm_weights = layernorm_weights)

    assert np.allclose(out_simplegrad.val, out_pytorch.data.numpy(), rtol=1e-06, atol=1e-05)

    # Test backward pass
    out_simplegrad.backward() # Same as out.simplegrad.backward()
    # Also call backward on fork
    attn.backward() # otherwise the backward pass never gets initiated

    # floating point precision seems to create some flakiness
    # TODO: create tests with lower tolerance

    # compare gradient in weights from last layernorm
    assert np.allclose(layernorm_weights[-1].grad, pytorch_grad[-1]) # bias
    assert np.allclose(layernorm_weights[-2].grad, pytorch_grad[-2], atol=1e-05) # weight

    # compare gradients in weights from feedforward layer
    assert np.allclose(linear_weights[-1].grad, pytorch_grad[-5], atol=1e-05) # bias
    assert np.allclose(linear_weights[-2].grad, pytorch_grad[-6], atol=1e-05) # weight
    assert np.allclose(linear_weights[-3].grad, pytorch_grad[-7], atol=1e-05) # bias
    assert np.allclose(linear_weights[-4].grad, pytorch_grad[-8], atol=1e-05) # weight

    # compare gradient in weights from first layernorm
    assert np.allclose(layernorm_weights[-3].grad, pytorch_grad[-3], atol=1e-05) # bias
    assert np.allclose(layernorm_weights[-4].grad, pytorch_grad[-4], atol=1e-05) # weight

    # reshape in_proj_weights back
    in_proj_weight_grad = np.concatenate([w.grad.reshape(embed_dim, embed_dim) for w in in_proj_weights])
    # compare gradient from in_proj_weights
    assert np.allclose(in_proj_weight_grad, pytorch_grad[0], rtol=1e-07, atol=1e-06)

    # reshape out_proj_weights back
    out_proj_weight_grad = out_proj_weight.grad.reshape(embed_dim, embed_dim)
    assert np.allclose(out_proj_weight_grad, pytorch_grad[2], rtol=1e-07, atol=1e-05)



class Transformer:
    def __init__(self, embed_dim, num_heads, num_classes):
        assert embed_dim % num_heads == 0
        self.embed_dim, max_len = embed_dim, embed_dim*2
        head_dim = embed_dim // num_heads

        # Create matrix representing the positional encoding for max_len inputs
        pe = np.zeros((max_len, embed_dim))
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float), 1)
        div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(position * div_term) # even
        pe[:, 1::2] = np.cos(position * div_term) # odd
        self.pe = np.expand_dims(pe, 1)

        # initialize weights for input net
        self.w0 = Tensor(np.random.randn(embed_dim, num_classes))
        self.b0 = Tensor(np.random.randn(embed_dim))

        # initialize uniform weights for multihead-attention
        k = 1 / embed_dim
        self.query_dense = Tensor(np.random.uniform(size=(num_heads, head_dim, embed_dim), low=-np.sqrt(k), high=np.sqrt(k)))
        self.key_dense = Tensor(np.random.uniform(size=(num_heads, head_dim, embed_dim), low=-np.sqrt(k), high=np.sqrt(k)))
        self.value_dense = Tensor(np.random.uniform(size=(num_heads, head_dim, embed_dim), low=-np.sqrt(k), high=np.sqrt(k)))
        self.out_proj_weight = Tensor(np.random.uniform(size=(embed_dim, num_heads, head_dim), low=-np.sqrt(k), high=np.sqrt(k)))
        self.out_proj_bias = Tensor(np.zeros((num_heads, head_dim)))

        # initialize weights for normalization layer
        self.layernorm_weight = Tensor(np.ones(embed_dim))
        self.layernorm_bias = Tensor(np.zeros(embed_dim))

        # initialize weights for output net
        self.w1 = Tensor(np.random.randn(num_classes, embed_dim))
        self.b1 = Tensor(np.random.randn(num_classes))

    @property
    def params(self):
        return self.w0, self.b0, self.w1, self.b1, \
               self.query_dense, self.key_dense, self.value_dense, \
               self.out_proj_weight, self.out_proj_bias, \
               self.layernorm_weight, self.layernorm_bias

    def plot_attention(self):
        # plots the attention of the first batch and head
        attn = self.attention[0,0]
        plt.imshow(attn)
        plt.show()

    def forward(self, x: Tensor):
        """
        Inputs:
            x - Input features of shape [SeqLen, Batch, input_dim]
            note: input_dim is the same as num_categories
        """
        # map the one-hot input to a dense vector of shape [SeqLen, Batch, embed_dim]
        x.dot(self.w0, subscripts="ijk,lk->ijl").add(self.b0)

        # positional encoding for sequences
        x.add(Tensor(self.pe[:x.shape[0]]))

        # Encoder block
        residual = x.fork()
        ## Multi-Head Attention
        x, attn = self.multi_head_attention_forward(x)
        ## Add and Norm
        x.add(residual).layer_norm(-1, self.layernorm_weight, self.layernorm_bias)

        # Output net maps the vector back to a sparse one-hot output
        return x.dot(self.w1, subscripts="ijk,lk->ijl").add(self.b1)

    def multi_head_attention_forward(self, x, mask=None):
        k = x.fork().dot(self.key_dense, subscripts="ijk,lmk->jlim")
        v = x.fork().dot(self.value_dense, subscripts="ijk,lmk->jlim")
        q = x.dot(self.query_dense, subscripts="ijk,lmk->jlim")

        attn_logits = q.dot(k, subscripts="ijkl,ijml->ijkm").div(Tensor(q.shape[-1]).pow(Tensor(0.5)))

        if mask:
            attn_logits.add(mask)
        attn = attn_logits.softmax()
        self.attention = attn.val

        values = attn.dot(v, subscripts="ijkl,ijlm->ijkm")
        output = values.dot(self.out_proj_weight, subscripts="ijkl,mjl->kim").add(self.out_proj_bias)
        return output, attn

def test_seq2seq_model():
    # taken from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Sequence-to-Sequence
    # Given a sequence of N numbers, the task is to reverse the input sequence.
    # RNNs can have issues with such because the task requires long-term dependencies.
    # Transformers are expected to excel on this task.

    # Data hyperparameters
    seq_len = 16
    batches = 400
    batch_size = 64
    num_categories = 10

    # training data
    X = np.random.randint(num_categories, size=(batches, seq_len, batch_size))
    y = X[:,::-1,:]

    # one-hot encode training data
    X = np.eye(num_categories)[X].astype(np.float32)
    y = np.eye(num_categories)[y].astype(np.float32)
    batches = zip(X, y)

    # Model hyperparameters
    embed_dim = 32
    num_heads = 1

    model = Transformer(embed_dim, num_heads, num_categories)
    optim = Adam(model.params, learning_rate=5e-4)

    for epoch, batch in it.product(range(epochs := 10), batches):
        X, y = batch

        out = model.forward(Tensor(X))
        # compute accuracy
        acc = (out.val.argmax(axis=-1) == y.argmax(axis=-1)).mean()

        # categorical prediction since output is one-hot
        out = out.logsoftmax()
        # negative log likelihood loss
        loss = Tensor(y).mul(out).mul(Tensor(-1.0)).sum(axis=-1).mean(axis=(0,1))

        optim.zero_grad()
        loss.backward()
        optim.step()

    # generate some test data
    X = np.random.randint(num_categories, size=(seq_len, batch_size))
    y = X[::-1,:]
    X = np.eye(num_categories)[X]

    # check accuracy for test data
    out = model.forward(Tensor(X))
    acc = (out.val.argmax(axis=-1) == y).astype(np.float32).mean()

    assert acc == 1.0

    # plot attention map
    # It should resemble a diagonal line, since each
    # token attends to the token on the flipside
    model.plot_attention()
