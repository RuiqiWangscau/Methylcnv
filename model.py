"""Inference-only model definition extracted from the supplied source."""
import torch
import torch.nn as nn

class BiLSTM_CNV(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(BiLSTM_CNV, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(2)
        context = lstm_out * attn_weights
        context = self.dropout(context)
        out = self.fc(context)
        return out
