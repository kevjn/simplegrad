from simplegrad import Tensor, Adam
import numpy as np
np.random.seed(1337)

import torch
import matplotlib.pyplot as plt

class LSTM():

    def __init__(self, input_size, hidden_dim, seq_len):
            self.input_dim = input_size
            self.hidden_dim = hidden_dim
            self.seq_len = seq_len

            k = np.sqrt(1/hidden_dim)

            self.w = [Tensor(np.random.uniform(-k, k, size=(hidden_dim, input_size))) for _ in range(4)]
            self.wb = [Tensor(np.random.uniform(-k, k, size=(hidden_dim, 1))) for _ in range(4)]

            self.u = [Tensor(np.random.uniform(-k, k, size=(hidden_dim, hidden_dim))) for _ in range(4)]
            self.ub = [Tensor(np.random.uniform(-k, k, size=(hidden_dim, 1))) for _ in range(4)]

            self.gates = Tensor.tanh, Tensor.sigmoid, Tensor.sigmoid, Tensor.sigmoid

    def init_hidden(self):
        return Tensor(np.zeros((1, 1, self.hidden_dim))), \
                Tensor(np.zeros((1, 1, self.hidden_dim)))

    def forward(self, seq):
        
        prev_ht, prev_ct = self.init_hidden()

        for t in range(len(seq)):
            inp = Tensor(seq[t])
            prev_ht, prev_ct = self.cell_forward(inp, prev_ht, prev_ct)

        # in reality the model should probably return output for all timesteps, but since
        # tensor concatenation is not supported yet just return output from the last timestep.
        return prev_ht

        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        return lstm_out.view(len(seq), -1)

    def cell_forward(self, X, prev_ht, prev_ct):
        at, it, ft, ot = (
            gate(X.fork().mul(w).add(wb).add(prev_ht.fork().mul(u).add(ub)))
            for gate, w, wb, u, ub in zip(self.gates, self.w, self.wb, self.u, self.ub)
        )

        ct = ft.mul(prev_ct).add(it.mul(at))
        ht = ot.mul(ct.tanh())

        return ht, ct

    @property
    def params(self):
        return *self.w, *self.wb, *self.u, *self.ub

def test_predict_sine_wave():
    # parameters
    INPUT_LEN = 40
    SEQ_LEN = 20

    seq = np.sin(np.arange(0, INPUT_LEN, 0.01))
    training_data = [(seq[i:i+SEQ_LEN], seq[i+1:i+1+SEQ_LEN]) for i in range(len(seq)-SEQ_LEN-1)]

    model = LSTM(1,1,SEQ_LEN)
    optim = Adam(model.params)

    # training
    for epoch in range(3):
        for seq, outs in training_data[:700]:
            optim.zero_grad()
            
            model.hidden = model.init_hidden()
            modout = model.forward(seq)

            # Squared error loss
            loss = Tensor(outs[-1]).sub(modout).pow(Tensor(2)).sum()
            loss.backward()

            optim.step()
    
    assert loss.val < 0.01

    # testing
    y_pred = list(training_data[700:][0][0])
    y_true = []
    for (seq, trueVal) in training_data[700:]:
        pred = model.forward(seq).val

        y_pred.append(pred[-1].ravel()[0])
        y_true.append(trueVal[-1])
    
    # plot result
    plt.plot(y_pred, label="prediction")
    plt.plot(y_true, label="ground truth")
    plt.legend(loc="upper right")
    plt.show()