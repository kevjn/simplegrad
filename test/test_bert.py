import pytest
import torch
from transformers import BertTokenizer
from simplegrad import Tensor
import numpy as np

@pytest.mark.parametrize('text1', ['The quick brown fox jumps over the lazy dog', 'Test'])
@pytest.mark.parametrize('text2', ['Hello world', 'Hello world again'])
def test_wordpiece_embedding_layer(text1, text2):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

    # BERT has a maximum input length of 512.
    inputs = tokenizer.encode_plus(text1, text2, return_tensors='pt')

    # ==== pytorch ====
    out_pt = model.embeddings(inputs['input_ids'], inputs['token_type_ids'])

    inputs = tokenizer.encode_plus(text1, text2, return_tensors='np')

    params = model.embeddings.parameters()
    # ==== simplegrad =====
    word_embeddings_weight = Tensor(next(params).detach().numpy(), '_')
    position_embeddings_weight = Tensor(next(params).detach().numpy(), '_')
    token_type_embeddings_weight = Tensor(next(params).detach().numpy(), '_')
    layernorm_weight = Tensor(next(params).detach().numpy(), '_')
    layernorm_bias = Tensor(next(params).detach().numpy(), '_')

    assert next(params, None) is None # make sure generator is exhausted

    # embedding forward
    x = word_embeddings_weight[inputs['input_ids']]

    y = position_embeddings_weight[:inputs['input_ids'].shape[1]]

    z = token_type_embeddings_weight[inputs['token_type_ids']]

    out = x.add(y).add(z)

    out_sg = out.layer_norm(-1, layernorm_weight, layernorm_bias, 1e-12)

    assert np.allclose(out_pt.data.numpy(), out_sg.data.view(np.ndarray), atol=1e-07)

    out_pt.sum().backward()
    out_sg.sum().backward()

    assert np.allclose(model.embeddings.word_embeddings.weight.grad.numpy(), 
                       word_embeddings_weight.grad.view(np.ndarray), atol=1e-04)

    assert np.allclose(model.embeddings.position_embeddings.weight.grad.numpy(), 
                       position_embeddings_weight.grad.view(np.ndarray), atol=1e-04)

    assert np.allclose(model.embeddings.token_type_embeddings.weight.grad.numpy(), 
                       token_type_embeddings_weight.grad.view(np.ndarray), atol=1e-04)

class BERT:
    """BERT-base model from a pretained pytorch model"""

    def __init__(self, torch_model):
        self.embedding_params = tuple (
            Tensor(w.detach().numpy()) for w in torch_model.embeddings.parameters()
        )
        # 5 parameters in embedding layer
        assert len(self.embedding_params) == 5

        self.encoder_params = tuple (
            tuple (
                Tensor(w.detach().numpy()) for w in it.chain ( 
                    *((w.reshape(12, 64, 768) , b.reshape(1,12,1,64)) for w,b in zip(layer[:6:2], layer[1:6:2])),
                     (layer[6].reshape(768, 12, 64), layer[7].reshape(-1)), 
                     layer[8:] )
            ) for layer in [[*x.parameters()] for x in torch_model.encoder.layer]
        )

        # 12 multi-layers transformer blocks
        assert len(self.encoder_params) == 12
        # 16 parameters in each layer
        assert len(self.encoder_params[0]) == 16

        self.pooler_params = tuple (
            Tensor(w.detach().numpy()) for w in torch_model.pooler.parameters()
        )

        # 2 parameters in pooler (output) layer
        assert len(self.pooler_params) == 2

    def forward(self, input_ids: np.ndarray, token_type_ids: np.ndarray, attention_mask=None):
        # embedding block
        token, position, segment, weight, bias = self.embedding_params
        x = token[input_ids].add(position[:input_ids.shape[1]]).add(segment[token_type_ids])
        x = x.layer_norm(-1, weight, bias, 1e-12) # layer normalization

        # first transformer block
        for layer in [self.encoder_params[0]]:
            residual = x.fork()
            # multi-head self-attention block
            k = x.fork().einsum("ijk,lmk->jlim", layer[2]).add(layer[3])
            v = x.fork().einsum("ijk,lmk->jlim", layer[4]).add(layer[5])
            q = x.einsum("ijk,lmk->jlim", layer[0]).add(layer[1])

            attn_logits = q.einsum("ijkl,mjkl->kjim", k).div(Tensor(64).pow(Tensor(0.5)))
            attn = attn_logits.softmax()

            values = attn.einsum("ijkl,ljim->ijkm", v)
            x = values.einsum("ijkl,mjl->ikm", layer[6]).add(layer[7])

            # add and norm
            x.add(residual).layer_norm(-1, layer[8], layer[9], 1e-12)

            residual = x.fork()

            # intermediate block
            x = GELU(x.einsum("ijk,lk->ijl", layer[10]).add(layer[11]))

            # output block
            x = x.einsum("ijk,lk->ijl", layer[12]).add(layer[13])
            x = x.add(residual).layer_norm(-1, layer[14], layer[15], 1e-12)

        return x

def GELU(x):
    lhs = Tensor(np.sqrt(2.0 / np.pi))
    rhs = x.fork().pow(Tensor(3)).mul(Tensor(0.044715)).add(x.fork())
    inner = lhs.mul(rhs).tanh()
    return x.mul(Tensor(0.5)).mul(inner.add(Tensor(1.0)))

def test_GELU():
    x = np.random.randn(1, 5, 3072)

    out_sg = GELU(x_sg := Tensor(x))
    out_pt = torch.nn.functional.gelu(x_pt := torch.tensor(x, requires_grad=True))
    assert np.allclose(out_sg.data.view(np.ndarray), out_pt.data.numpy(), atol=1e-3)

    out_sg.sum().backward()
    out_pt.sum().backward()

    assert np.allclose(x_sg.grad.view(np.ndarray), x_pt.grad.numpy(), atol=1e-3)

def test_bert_base():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    torch_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', hidden_dropout_prob=0.0)
    torch_model.eval() # set model to inference mode

    model = BERT(torch_model)

    # ==== pytorch ====
    inputs = tokenizer.encode_plus("Hello", "World", return_tensors='pt', return_attention_mask=False)
    out_embedding = torch_model.embeddings(**inputs)
    out_pt = torch_model.encoder.layer[0](out_embedding)[0]

    # ==== simplegrad ====
    inputs = tokenizer.encode_plus("Hello", "World", return_tensors='np')
    out_sg = model.forward(**inputs)

    np.allclose(out_sg.data.view(np.ndarray), out_pt.data.numpy(), atol=0.01)