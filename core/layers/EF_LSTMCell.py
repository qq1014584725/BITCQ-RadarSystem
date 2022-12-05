import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, inChannel, hidden, kernel, shape):
        super(ConvLSTMCell, self).__init__()
        self.hidden = hidden
        self.width = shape
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=inChannel,
            out_channels=4*hidden,
            kernel_size=kernel,
            padding=kernel // 2),
            nn.LayerNorm([4*hidden, self.width, self.width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=hidden,
            out_channels=4*hidden,
            kernel_size=kernel,
            padding=kernel // 2),
            nn.LayerNorm([4*hidden, self.width, self.width])
        )

    def forward(self, input, state):
        h, c = state
        x_concat = self.conv_x(input)
        h_concat = self.conv_h(h)

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + c_new)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new
