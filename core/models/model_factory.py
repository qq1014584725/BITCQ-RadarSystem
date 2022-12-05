

from math import fabs
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import predict

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'convlstm':predict.ConvLSTM,
            'ef_convlstm':predict.EFConvLSTM,
            'predrnn':predict.PredRNN,
            'ef_prernn':predict.EFPredRNN,
            'predrnn_plus': predict.PredRNN_Plus,
            'interact_convlstm': predict.InteractionConvLSTM,
            'interact_predrnn':predict.InteractionPredRNN,
            'interact_predrnn_plus':predict.InteractionPredRNN_Plus,
            'cst_predrnn':predict.CST_PredRNN,
            'sst_predrnn': predict.SST_PredRNN,
            'dst_predrnn':predict.DST_PredRNN,
            'interact_dst_predrnn': predict.InteractionDST_PredRNN,
            'traj_gru' : predict.TrajGRU,
            'pfst_convlstm' : predict.EFPFSTLSTM,
            'ef_mim' : predict.MIM,
        }

        if configs.model_name in networks_map:

            Network = networks_map[configs.model_name]
            # self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
            if self.configs.device == 'cpu':
                self.network = Network(self.num_layers, self.num_hidden, configs)
            else:
                self.network = Network(self.num_layers, self.num_hidden, configs).cuda()
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        if self.configs.is_parallel:
            self.network = nn.DataParallel(self.network)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss()
        self.MAE_criterion = nn.L1Loss()


    def save(self,ite = None):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        if ite == None:
            checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        else:
            checkpoint_path = os.path.join(self.configs.save_dir, 'model_'+str(ite)+'.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        if self.configs.device == 'cpu':
            stats = torch.load(checkpoint_path, map_location='cpu')
        else:
            stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])
        print('model has been loaded')

    def train(self, frames, mask):
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        frames_tensor = torch.FloatTensor(frames).cuda()
        mask_tensor = torch.FloatTensor(mask).cuda()
        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor)
        if self.configs.model_type == 'ef':
            loss = self.MSE_criterion(next_frames, frames_tensor[:, self.configs.input_length:])+\
                self.MAE_criterion(next_frames, frames_tensor[:, self.configs.input_length:])
        elif self.configs.model_type == 'seq':
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
                self.MAE_criterion(next_frames, frames_tensor[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).cuda()
        mask_tensor = torch.FloatTensor(mask).cuda()
        next_frames = self.network(frames_tensor, mask_tensor)
        # if self.configs.model_type == 'ef':
        #     loss = self.MSE_criterion(next_frames, frames_tensor[:, self.configs.input_length:])+\
        #     self.MAE_criterion(next_frames, frames_tensor[:, self.configs.input_length:])
        # else:
        #     loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])+\
        #     self.MAE_criterion(next_frames, frames_tensor[:, 1:])
        #        # + 0.02 * self.SSIM_criterion(next_frames, frames_tensor[:, 1:])

        return next_frames.detach().cpu().numpy(), False
    
    def use(self, frames, mask):
        if self.configs.device == "cpu":
            frames_tensor = torch.FloatTensor(frames)
            mask_tensor = torch.FloatTensor(mask)
        else:
            frames_tensor = torch.FloatTensor(frames).cuda()
            mask_tensor = torch.FloatTensor(mask).cuda()
            
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy(), False