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
    out_pt = model.embeddings(inputs['input_ids'], inputs['token_type_ids']).data.numpy()

    params = model.embeddings.parameters()
    # ==== simplegrad =====
    word_embeddings_weight = next(params).detach().numpy()
    position_embeddings_weight = next(params).detach().numpy()
    token_type_embeddings_weight = next(params).detach().numpy()
    layernorm_weight = Tensor(next(params).detach().numpy(), '_')
    layernorm_bias = Tensor(next(params).detach().numpy(), '_')

    assert next(params, None) is None # make sure generator is exhausted

    # embedding forward
    x = word_embeddings_weight[inputs['input_ids']]

    y = position_embeddings_weight[:inputs['input_ids'].size(1)]

    z = token_type_embeddings_weight[inputs['token_type_ids']]

    out = x + y + z

    out = Tensor(out).layer_norm(-1, layernorm_weight, layernorm_bias, 1e-12)
    out_sg = out.data.view(np.ndarray)

    assert np.allclose(out_pt, out_sg, atol=1e-07)

def test_bert_base():
    # TODO: 12 encoder layers (transformer blocks)
    return