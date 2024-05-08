import torch.nn as nn
import torch.nn.functional as F

import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, in_dims, hidden_dims, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dims = hidden_dims
        padding = kernel_size // 2

        # creating one conv for all matrix generation
        self.conv = nn.Conv2d(
            in_dims + hidden_dims, hidden_dims * 4, kernel_size, padding=padding
        )  # includes bias

    def forward(self, x_t, prev):
        h_t, c_t = prev

        # concatenating input and hidden by channel dimension
        x = torch.cat([x_t, h_t], dim=1)
        xh_i, xh_f, xh_c, xh_o = torch.split(self.conv(x), self.hidden_dims, dim=1)

        # forget gate
        f = F.sigmoid(xh_f)

        # update cell state
        c = c_t * f

        # input gate
        i = F.sigmoid(xh_i)

        # new candidate
        c_candidate = F.tanh(xh_c)

        # update cell state
        c = c + i * c_candidate

        # output gate
        o = F.sigmoid(xh_o)

        # update hidden state
        h = o * F.tanh(c)

        return (h, c)


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_dims, hidden_dims, kernel_size):
        super(ConvLSTMLayer, self).__init__()
        self.hidden_dims = hidden_dims
        self.conv_lstm = ConvLSTMCell(in_dims, hidden_dims, kernel_size)

    def forward(self, x):
        b, t, c, h, w = x.shape

        # initializing tensor for initial hidden and cell states
        zeros = torch.zeros(b, self.hidden_dims, h, w)
        h_t, c_t = zeros.to(x.device), zeros.to(x.device)

        # record hidden_states
        hidden_states = torch.zeros(t, b, self.hidden_dims, h, w).to(x.device)

        # unroll ConvLSTMCell over time
        for i in range(t):
            h_t, c_t = self.conv_lstm(x[:, i, :, :, :], (h_t, c_t))
            hidden_states[i] = h_t

        return hidden_states.permute(1, 0, 2, 3, 4)
