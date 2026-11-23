"""RNN / LSTM / BiLSTM / attention written from the assignment equations.

No nn.RNN, nn.LSTM, nn.GRU or ready-made attention modules.
"""

import math
import torch
import torch.nn as nn


def _uniform_(param, hidden_size):
    bound = 1.0 / math.sqrt(hidden_size)
    nn.init.uniform_(param, -bound, bound)


class RNNCell(nn.Module):
    """ht = tanh(W xt + U h{t-1} + b). Not used in HARNet; kept for Task 6.6."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(hidden_size))
        _uniform_(self.W, hidden_size)
        _uniform_(self.U, hidden_size)
        _uniform_(self.b, hidden_size)

    def forward(self, x, h):
        # x: (B, D), h: (B, H)
        return torch.tanh(x @ self.W.T + h @ self.U.T + self.b)


class LSTMCell(nn.Module):
    """Four gates packed in W (4H x D), U (4H x H), b (4H).
    Order: input, forget, cell candidate, output.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(4 * hidden_size))
        _uniform_(self.W, hidden_size)
        _uniform_(self.U, hidden_size)
        _uniform_(self.b, hidden_size)

    def forward(self, x, h, c):
        # x: (B, D), h/c: (B, H)
        gates = x @ self.W.T + h @ self.U.T + self.b
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTM(nn.Module):
    """Run the cell over t = 1..T. Returns all hidden states, shape (B, T, H)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (B, T, D)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_size)
        c = x.new_zeros(B, self.hidden_size)
        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class BiLSTM(nn.Module):
    """Forward LSTM on x1..xT, backward LSTM on xT..x1, then reverse the
    backward outputs so position t lines up with xt. Concatenate -> (B, T, 2H).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fwd = LSTM(input_size, hidden_size)
        self.bwd = LSTM(input_size, hidden_size)

    def forward(self, x):
        h_fwd = self.fwd(x)
        h_bwd = self.bwd(torch.flip(x, dims=[1]))
        h_bwd = torch.flip(h_bwd, dims=[1])
        return torch.cat([h_fwd, h_bwd], dim=2)


class Attention(nn.Module):
    """et = v^T tanh(Wa ht + ba), alpha = softmax(e), c = sum alpha_t ht."""

    def __init__(self, hidden_in, attn_size=64):
        super().__init__()
        self.Wa = nn.Parameter(torch.empty(attn_size, hidden_in))
        self.ba = nn.Parameter(torch.empty(attn_size))
        self.v = nn.Parameter(torch.empty(attn_size))
        _uniform_(self.Wa, attn_size)
        _uniform_(self.ba, attn_size)
        _uniform_(self.v, attn_size)

    def forward(self, h):
        # h: (B, T, 2H)
        e = torch.tanh(h @ self.Wa.T + self.ba) @ self.v  # (B, T)
        alpha = torch.softmax(e, dim=1)
        context = (alpha.unsqueeze(-1) * h).sum(dim=1)
        return context, alpha
