import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # Parameters
        self.input_size = input_size  # Feature size
        self.hidden_size = hidden_size  # Number of hidden units
        self.num_layers = num_layers  # Number of LSTM layers to stack
        self.output_size = output_size  # Number of output
        self.num_directions = 1
        self.n_outputs = 1
        self.bias = True
        self.batch_first = True
        self.dropout = 0
        self.bidirectional = False
        # LSTM Layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional)
        # Fully Connected Layers
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size) for i in range(self.n_outputs)]
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # Activation Function
        self.act = nn.Sigmoid()

    def forward(self, x):
        # Hidden units
        h0 = torch.randn(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device=x.device)
        # Memory units
        c0 = torch.randn(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # preds = [self.fcs[i](out[:, -1, :]) for i in range(self.n_outputs)]
        out = self.fc(out[:, -1, :])
        pred = self.act(out)
        return pred
