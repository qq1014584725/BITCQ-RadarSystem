from textwrap import wrap
import torch
import torch.nn as nn


class PFST_ConvLSTMCell(nn.Module):
    def __init__(self,in_channel,hidden,kernel,shape,device):
        super(PFST_ConvLSTMCell, self).__init__()
        self.device = device
        self.hidden = hidden
        self.width = shape
        self._forget_bias = 1.0
        self.conv_x1_pre = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
            out_channels = 2,
            kernel_size = kernel,
            padding = kernel//2),
            nn.LayerNorm([2, self.width, self.width])
        )
        self.conv_x2_pre = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
            out_channels = 2,
            kernel_size = kernel,
            padding = kernel//2),
            nn.LayerNorm([2, self.width, self.width])
        )
        self.conv_h_pre = nn.Sequential(
            nn.Conv2d(in_channels=hidden,
            out_channels = 2,
            kernel_size=kernel,
            padding=kernel//2),
            nn.LayerNorm([2, self.width, self.width])
        )
        self.conv_m_pre = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
            out_channels = 2,
            kernel_size=kernel,
            padding=kernel//2),
            nn.LayerNorm([2, self.width, self.width])
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
            out_channels = 7*hidden,
            kernel_size = kernel,
            padding = kernel//2),
            nn.LayerNorm([7*hidden, self.width, self.width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels=hidden,
            out_channels=4*hidden,
            kernel_size=kernel,
            padding=kernel//2),
            nn.LayerNorm([4*hidden, self.width, self.width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
            out_channels=3*hidden,
            kernel_size=kernel,
            padding=kernel//2),
            nn.LayerNorm([3*hidden, self.width, self.width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=2*hidden,
            out_channels=hidden,
            kernel_size=kernel,
            padding=kernel//2),
            nn.LayerNorm([hidden, self.width, self.width])
        )
        self.conv_last = nn.Conv2d(2*hidden,hidden,kernel_size=1,stride=1,padding=0)

    def wrap(self,input, flow):
        
        B, C, H, W = input.size()
        # mesh grid
        if torch.cuda.is_available() & (self.device != "cpu"):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
        else:
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(input, vgrid)
        return output

    def forward(self, pre_input, input, state):
        h,c,m = state
        
        x1_pre = self.conv_x1_pre(pre_input)
        x2_pre = self.conv_x2_pre(pre_input)
        h_pre = self.conv_h_pre(h)
        m_pre = self.conv_m_pre(m)
        
        d_t = x1_pre + x2_pre + h_pre + m_pre
        
        h = self.wrap(h, d_t)
        c = self.wrap(c, d_t)
        
        x_concat = self.conv_x(input)
        m_concat = self.conv_m(m)
        h_concat = self.conv_h(h)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.hidden, dim=1)

        i_t = torch.sigmoid(i_x+i_h)
        f_t = torch.sigmoid(f_x+f_h+self._forget_bias)
        g_t = torch.tanh(g_x+g_h)

        c_new = f_t*c+i_t*g_t

        i_t_prime = torch.sigmoid(i_x_prime+i_m)
        f_t_prime = torch.sigmoid(f_x_prime+f_m+self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime+g_m)

        m_new = f_t_prime*m+i_t_prime*g_t_prime
        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x+o_h+self.conv_o(mem))
        h_new = o_t*torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new