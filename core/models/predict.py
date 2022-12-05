
__author__ = 'chuyao'
from tkinter import W
import torch
import torch.nn as nn
import copy

from core.layers.LSTMCell import LSTMCell
from core.layers.ST_LSTMCell import SpatioTemporalLSTMCell, ST_LSTMCell
from core.layers.CausalLSTMCell import CausalLSTMCell
from core.layers.GradientHighwayUnit import GHU

from core.layers.InterLSTMCell import InterLSTMCell
from core.layers.InterST_LSTMCell import InterSpatioTemporalLSTMCell
from core.layers.InteractCausalLSTMCell import InteractCausalLSTMCell

from core.layers.CST_LSTMCell import CST_LSTMCell
from core.layers.SST_LSTMCell import SST_LSTMCell
from core.layers.DST_LSTMCell import DST_LSTMCell
from core.layers.InterDST_LSTMCell import InterDST_LSTMCell
from core.layers.EF_LSTMCell import ConvLSTMCell
from core.layers.Traj_GRUCell import TrajGRUCell
from core.layers.PFST_ConvLSTM import PFST_ConvLSTMCell
from core.layers.MIMCell import MIMCell, MIMS


class ConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(ConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)


        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames


class EFConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(EFConvLSTM, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.total_len = configs.total_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=3, padding=1),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),                        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=3, padding=1, output_padding=0)
            )
        
        self.decoder.add_module("last_activation", nn.Sigmoid())
        
        self.convlstm_num = num_layers
        self.convlstm_in_c = num_hidden
        self.convlstm_out_c = num_hidden
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(ConvLSTMCell(inChannel=self.convlstm_in_c[layer_i],
                                                  hidden=self.convlstm_out_c[layer_i],
                                                  kernel=configs.kernel,
                                                  shape = 32))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)


    def forward(self, short_x, mask_x):
        short_x = short_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        mask_x = mask_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]
        height = short_x.size()[3]//12
        width = short_x.size()[4]//12
        out_len = self.total_len - input_len

        # motion context-aware video prediction
        h, c, out_pred = [], [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], height, width).to(short_x.device)#self.device
            h.append(zero_state)
            c.append(zero_state)

        for seq_i in range(self.total_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:
                    h[layer_i], c[layer_i] = self.convlstm_list[layer_i](input_x, (h[layer_i], c[layer_i]))
                else:
                    h[layer_i], c[layer_i] = self.convlstm_list[layer_i](h[layer_i-1], (h[layer_i], c[layer_i]))

            x_gen = h[self.convlstm_num-1]
            x_gen = self.decoder(x_gen)

            if seq_i >= input_len-1:
                out_pred.append(x_gen)

                # attention = self.attention_func(torch.cat([c[-1], memory_feature], dim=1))
                # attention = torch.reshape(attention, (-1, self.attention_size, 1, 1))
                # memory_feature_att = memory_feature * attention
                # out_pred.append(self.decoder(torch.cat([h[-1], memory_feature_att], dim=1)))

        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :].permute(0, 1, 3, 4, 2).contiguous()

        return out_pred


class TrajGRU(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(TrajGRU, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.total_len = configs.total_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=3, padding=1),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True)
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=3, padding=1, output_padding=0)
            )
        
        self.decoder.add_module("last_activation", nn.Sigmoid())

        # if args.dataset == 'kth':
        #     self.decoder.add_module("last_activation", nn.Sigmoid())

        self.convlstm_in_c = num_hidden
        self.convlstm_num  = num_layers
        self.convlstm_out_c = num_hidden
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(TrajGRUCell(inChannel=self.convlstm_in_c[layer_i],
                                                  hidden=self.convlstm_out_c[layer_i],
                                                  kernel=configs.kernel,
                                                  ))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)


    def forward(self, short_x, mask_x):
        short_x = short_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        mask_x = mask_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]
        height = short_x.size()[3]//12
        width = short_x.size()[4]//12
        out_len = self.total_len - input_len

        # motion context-aware video prediction
        h, out_pred = [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], height, width).to(short_x.device)#self.device
            h.append(zero_state)

        for seq_i in range(self.total_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:
                    h[layer_i] = self.convlstm_list[layer_i](input_x, h[layer_i])
                else:
                    h[layer_i] = self.convlstm_list[layer_i](h[layer_i-1], h[layer_i])

            x_gen = h[self.convlstm_num-1]
            x_gen = self.decoder(x_gen)

            if seq_i >= input_len-1:
                out_pred.append(x_gen)


        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :].permute(0, 1, 3, 4, 2).contiguous()

        return out_pred


class EFPredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(EFPredRNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.total_len = configs.total_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=3, padding=1),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),                        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=3, padding=1, output_padding=0)
            )
        
        self.decoder.add_module("last_activation", nn.Sigmoid())


        self.convlstm_in_c = num_hidden
        self.convlstm_num  = num_layers
        self.convlstm_out_c = num_hidden
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(ST_LSTMCell(in_channel=self.convlstm_in_c[layer_i],
                                                  hidden=self.convlstm_out_c[layer_i],
                                                  kernel=configs.kernel,
                                                  shape = 32))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)

    def forward(self, short_x, mask_x):
        short_x = short_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        mask_x = mask_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]
        height = short_x.size()[3]//12  # //总的stride,原图下采样12倍
        width = short_x.size()[4]//12
        out_len = self.total_len - input_len

        # motion context-aware video prediction
        h, c, out_pred = [], [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], height, width).to(short_x.device)#self.device
            h.append(zero_state)
            c.append(zero_state)
        memory = torch.zeros([batch_size, self.convlstm_in_c[0],height,width]).to(short_x.device)

        for seq_i in range(self.total_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:
                    h[layer_i], c[layer_i], memory = self.convlstm_list[layer_i](input_x, (h[layer_i], c[layer_i], memory))
                else:
                    h[layer_i], c[layer_i], memory = self.convlstm_list[layer_i](h[layer_i-1], (h[layer_i], c[layer_i], memory))

            x_gen = h[self.convlstm_num-1]
            x_gen = self.decoder(x_gen)

            if seq_i >= input_len-1:
                out_pred.append(x_gen)

        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :].permute(0, 1, 3, 4, 2).contiguous()

        return out_pred


class EFPFSTLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(EFPFSTLSTM, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.total_len = configs.total_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=3, padding=1),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),                        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            # nn.GroupNorm(16,256),
            # nn.ELU(inplace=True)
            )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            # nn.GroupNorm(8,128),
            # nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=True),            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=3, padding=1, output_padding=0)
            )
        
        self.decoder.add_module("last_activation", nn.Sigmoid())


        self.convlstm_in_c = num_hidden
        self.convlstm_num  = num_layers
        self.convlstm_out_c = num_hidden
        self.convlstm_list = []
        for layer_i in range(self.convlstm_num):
            self.convlstm_list.append(PFST_ConvLSTMCell(in_channel=self.convlstm_in_c[layer_i],
                                                        hidden=self.convlstm_out_c[layer_i],
                                                        kernel=configs.kernel,
                                                        shape = 32,
                                                        device = self.configs.device))
        self.convlstm_list = nn.ModuleList(self.convlstm_list)

    def forward(self, short_x, mask_x):
        short_x = short_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        mask_x = mask_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]
        height = short_x.size()[3]//12  # //总的stride,原图下采样12倍
        width = short_x.size()[4]//12
        out_len = self.total_len - input_len

        # motion context-aware video prediction
        h, c, out_pred = [], [], []
        for layer_i in range(self.convlstm_num):
            zero_state = torch.zeros(batch_size, self.convlstm_in_c[layer_i], height, width).to(short_x.device)#self.device
            h.append(zero_state)
            c.append(zero_state)
        memory = torch.zeros([batch_size, self.convlstm_in_c[0],height,width]).to(short_x.device)
        x0 = torch.zeros([batch_size, self.convlstm_in_c[0],height,width]).to(short_x.device)
        
        for seq_i in range(self.total_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            for layer_i in range(self.convlstm_num):
                if layer_i == 0:
                    h[layer_i], c[layer_i], memory = self.convlstm_list[layer_i](x0, input_x, (h[layer_i], c[layer_i], memory))
                else:
                    h[layer_i], c[layer_i], memory = self.convlstm_list[layer_i](x0 ,h[layer_i-1], (h[layer_i], c[layer_i], memory))

            x0 = input_x
            
            x_gen = h[self.convlstm_num-1]
            x_gen = self.decoder(x_gen)

            if seq_i >= input_len-1:
                out_pred.append(x_gen)

        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :].permute(0, 1, 3, 4, 2).contiguous()

        return out_pred


class MIM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(MIM, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.total_len = configs.total_length
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=3, padding=1),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=False),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=False),            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=False),                        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.GroupNorm(8,128),
            nn.ELU(inplace=False))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(4,64),
            nn.ELU(inplace=False),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(2,32),
            nn.ELU(inplace=False),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1,16),
            nn.ELU(inplace=False),            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, output_padding=0),
            nn.GroupNorm(1,8),
            nn.ELU(inplace=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), stride=3, padding=1, output_padding=0)
            )
        
        self.decoder.add_module("last_activation", nn.Sigmoid())


        self.convlstm_in_c = num_hidden
        self.convlstm_num  = num_layers
        self.convlstm_out_c = num_hidden
        self.convlstm_list = []
        self.convlstm_list_diff = []
        for layer_i in range(self.convlstm_num):
            if layer_i < 1:
                self.convlstm_list.append(ST_LSTMCell(in_channel=self.convlstm_in_c[layer_i],
                                                  hidden=self.convlstm_out_c[layer_i],
                                                  kernel=configs.kernel,
                                                  shape = 32))
            else:
                self.convlstm_list.append(MIMCell(in_ch=self.convlstm_in_c[layer_i],
                                                out_ch=self.convlstm_out_c[layer_i],
                                                k_size=configs.kernel,
                                                h=32,
                                                w=32))
        
        for i in range(self.convlstm_num - 1):
            self.convlstm_list_diff.append(MIMS(self.convlstm_in_c[i + 1], 32, 32))
        
        self.convlstm_list = nn.ModuleList(self.convlstm_list)
        self.convlstm_list_diff = nn.ModuleList(self.convlstm_list_diff)

    def set_shape(self, shape, device):
        for i in range(1, self.convlstm_num):
            self.convlstm_list[i].set_shape(shape, device)

        for i in range(self.convlstm_num - 1):
            self.convlstm_list_diff[i].set_shape(shape, device)

    def forward(self, short_x, mask_x):
        if(self.configs.device != 'cpu'):
            self.set_shape([self.configs.batch_size, self.configs.img_channel, 32, 32], 'cuda')
        else:
            self.set_shape([self.configs.batch_size, self.configs.img_channel, 32, 32], 'cpu')
            
        short_x = short_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        mask_x = mask_x.permute(0, 1, 4, 2, 3)[:,0:10,:,:,:].contiguous()
        
        batch_size = short_x.size()[0]
        input_len= short_x.size()[1]
        height = short_x.size()[3]//12  # //总的stride,原图下采样12倍
        width = short_x.size()[4]//12
        out_len = self.total_len - input_len

        # motion context-aware video prediction
        st_memory = torch.zeros([batch_size, self.convlstm_in_c[0],height,width]).to(short_x.device)
        cell_state = [None] * self.convlstm_num
        hidden_state = [None] * self.convlstm_num
        cell_state_diff = [None] * (self.convlstm_num - 1)
        hidden_state_diff = [None] * (self.convlstm_num - 1)
        cell_state[0] = torch.zeros(batch_size, self.convlstm_in_c[0], height, width).to(short_x.device)
        hidden_state[0] = torch.zeros(batch_size, self.convlstm_in_c[0], height, width).to(short_x.device)
        
        out_pred = []
        
        for seq_i in range(self.total_len-1):
            if seq_i < input_len:
                input_x = short_x[:, seq_i, :, :, :]
                input_x = self.encoder(input_x)
            else:
                input_x = self.encoder(out_pred[-1])

            preh = hidden_state[0]
            
            hidden_state[0], cell_state[0], st_memory = self.convlstm_list[0](input_x, (hidden_state[0], cell_state[0], st_memory))
            
            for layer_i in range(1, self.convlstm_num):
                if seq_i == 0:
                     _, _ = self.convlstm_list_diff[layer_i-1](torch.zeros_like(hidden_state[layer_i-1]), None, None)
                else:
                    diff = hidden_state[layer_i-1] - preh if layer_i == 1 else hidden_state[layer_i-2]
                    hidden_state_diff[layer_i-1], cell_state_diff[layer_i-1] = self.convlstm_list_diff[layer_i-1](diff, hidden_state_diff[layer_i-1], cell_state_diff[layer_i-1])
                
                preh = hidden_state[layer_i]
                
                hidden_state[layer_i], cell_state[layer_i], st_memory = self.convlstm_list[layer_i](hidden_state[layer_i-1], hidden_state_diff[layer_i-1], hidden_state[layer_i], cell_state[layer_i], st_memory)
            
            x_gen = hidden_state[-1]
            x_gen = self.decoder(x_gen)

            if seq_i >= input_len-1:
                out_pred.append(x_gen)

        out_pred = torch.stack(out_pred)
        out_pred = out_pred.transpose(dim0=0, dim1=1)
        out_pred = out_pred[:, -out_len:, :, :, :].permute(0, 1, 3, 4, 2).contiguous()

        return out_pred


class PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames

class PredRNN_Plus(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN_Plus, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size

        self.gradient_highway = GHU(
            self.num_hidden[0],
            self.num_hidden[0],
            height,
            width,
            self.configs.filter_size,
            self.configs.stride,
        )
        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                CausalLSTMCell(in_channel, num_hidden_in, self.num_hidden[i], height,
                                       width, configs.filter_size,
                                       configs.stride,
                                       configs.layer_norm
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0], memory)

            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1],c_t[1], memory)

            for i in range(2, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],
                                                                memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames



class InteractionConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                InterLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm, configs.r)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)


        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames

class InteractionPredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionPredRNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(
                InterSpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm,configs.r)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)

        # memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()
        for t in range(self.configs.total_length-1):

            if t < self.configs.input_length:
                net = frames[:,t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        return next_frames

class InteractionPredRNN_Plus(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionPredRNN_Plus, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size

        self.gradient_highway = GHU(
            self.num_hidden[0],
            self.num_hidden[0],
            height,
            width,
            self.configs.filter_size,
            self.configs.stride,
        )
        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                InteractCausalLSTMCell(in_channel, num_hidden_in, self.num_hidden[i], height,
                                       width, configs.filter_size,
                                       configs.stride,
                                       configs.layer_norm,
                                       configs.r
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []

        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0], memory)

            z_t = self.gradient_highway(h_t[0], z_t)
            h_t[1], c_t[1], memory = self.cell_list[1](z_t, h_t[1],c_t[1], memory)

            for i in range(2, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],
                                                                memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames



class DST_PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(DST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                DST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(zeros.unsqueeze(1))
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = torch.cat([c_t_history[0],c_t[0].unsqueeze(1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames

class SST_PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(SST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                SST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(zeros.unsqueeze(1))
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = torch.cat([c_t_history[0],c_t[0].unsqueeze(1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames

class CST_PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(CST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                CST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(zeros.unsqueeze(1))
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = torch.cat([c_t_history[0],c_t[0].unsqueeze(1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames

class InteractionDST_PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(InteractionDST_PredRNN, self).__init__()


        self.configs = configs
        self.frame_channel = configs.img_channel * configs.patch_size * configs.patch_size
        wide_cell_list = []
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        cell_list = []

        width = configs.img_width // configs.patch_size
        height = configs.img_width // configs.patch_size


        for i in range(self.num_layers):
            num_hidden_in = self.num_hidden[i - 1]
            in_channel = self.frame_channel if i == 0 else num_hidden_in
            cell_list.append(
                InterDST_LSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                configs.stride, configs.layer_norm,configs.r
                               )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)





    def forward(self, frames, mask_true, is_training=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        h_t_wide = []
        c_t_wide = []
        c_t_history = []
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            # num_hidden_in = self.deep_num_hidden[i-1]
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            c_t_history.append(zeros.unsqueeze(1))
        memory = torch.zeros([batch, self.num_hidden[-1], height, width]).cuda()


        z_t = torch.zeros([batch, self.num_hidden[0], height,
                           width]).cuda()

        if is_training:
            seq_length = self.configs.total_length
        else:
            seq_length = self.configs.test_total_length

        for t in range(seq_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0],c_t[0],c_t_history[0], memory)
            c_t_history[0] = torch.cat([c_t_history[0],c_t[0].unsqueeze(1)],1)
            for i in range(1, self.num_layers):

                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1],
                                                                h_t[i], c_t[i],c_t_history[i],
                                                                memory)
                c_t_history[i] = torch.cat([c_t_history[i], c_t[i].unsqueeze(1)], 1)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4,
                                                              2).contiguous()

        return next_frames
    












