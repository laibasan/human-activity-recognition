"""HARNet: BiLSTM -> attention -> Linear(128, 6)."""

import torch
import torch.nn as nn
from layers import BiLSTM, Attention, RNNCell, LSTMCell


class HARNet(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, n_classes=6):
        super().__init__()
        self.bilstm = BiLSTM(input_size, hidden_size)
        self.attn = Attention(hidden_in=2 * hidden_size, attn_size=hidden_size)
        self.fc = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x):
        h = self.bilstm(x)  # (B, 128, 128)
        context, alpha = self.attn(h)
        logits = self.fc(context)
        return logits, alpha


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown():
    """Hand calculation that should add up to 46,982."""
    D, H = 9, 64
    lstm = 4 * H * D + 4 * H * H + 4 * H  # W, U, b
    bilstm = 2 * lstm
    attn = 64 * (2 * H) + 64 + 64  # Wa, ba, v
    linear = (2 * H) * 6 + 6
    total = bilstm + attn + linear
    return {
        "lstm_one_direction": lstm,
        "bilstm": bilstm,
        "attention": attn,
        "linear": linear,
        "total": total,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    net = HARNet()
    n = count_parameters(net)
    print("trainable parameters:", n)
    print("by hand:", param_breakdown())
    x = torch.randn(2, 128, 9)
    logits, alpha = net(x)
    print("logits", tuple(logits.shape), "alpha", tuple(alpha.shape))
    print("alpha sums", alpha.sum(dim=1).tolist())
    print("initial mean loss ~ log 6 =", float(torch.log(torch.tensor(6.0))))
    y = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(logits, y)
    print("fresh-weight loss on random batch:", float(loss))
