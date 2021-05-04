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


def test_bert_base():
    # TODO: 12 encoder layers (transformer blocks)
    return