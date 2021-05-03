import pytest
import torch
from transformers import BertTokenizer
from simplegrad import Tensor
import numpy as np

@pytest.mark.parametrize('text', 
                        ['The quick brown fox jumps over the lazy dog',
                         'Test'])
def test_wordpiece_embedding_layer(text):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

    # BERT has a maximum input length of 512.
    x = tokenizer.encode(text, return_tensors='pt')

    # ========== pytorch ==========
    out_pt = model.embeddings(x).data.numpy()

    params = model.embeddings.parameters()
    # ========= simplegrad =========
    word_embeddings_weight = next(params).detach().numpy()
    position_embeddings_weight = next(params).detach().numpy()
    token_type_embeddings_weight = next(params).detach().numpy()
    layernorm_weight = Tensor(next(params).detach().numpy(), '_')
    layernorm_bias = Tensor(next(params).detach().numpy(), '_')

    assert next(params, None) is None # make sure generator is exhausted

    # embedding forward
    indices = x
    segment_label = 0 # only use 1 segment for now

    x = word_embeddings_weight[indices]

    idx = np.broadcast_to(np.arange(indices.shape[1]), indices.shape)
    y = position_embeddings_weight[idx]

    idx = np.broadcast_to(segment_label, indices.shape)
    z = token_type_embeddings_weight[idx]

    out = x + y + z

    out = Tensor(out).layer_norm(-1, layernorm_weight, layernorm_bias, 1e-12)
    out_sg = out.data.view(np.ndarray)

    assert np.allclose(out_pt, out_sg, atol=1e-07)

def test_bert_base():
    # TODO: 12 encoder layers (transformer blocks)
    return