import torch
import torch.nn as nn
from torch.nn import functional as F


class TrajGRUCell(nn.Module):
    def __init__(self,inChannel, hidden, kernel, L=13):
        super(TrajGRUCell, self).__init__()
        self.convUV = nn.Sequential(*[
            nn.Conv2d(in_channels=inChannel+hidden, out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=L*2, kernel_size=(5,5), stride=1, padding=2)

        ])
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels=inChannel+hidden*L, out_channels=2*hidden, kernel_size=kernel,padding=kernel//2)
        self.rconvI = nn.Conv2d(in_channels=inChannel, out_channels=hidden,kernel_size=kernel,padding=kernel//2)
        self.rconvH = nn.Conv2d(in_channels=hidden*L, out_channels=hidden, kernel_size=kernel,padding=kernel//2)

    def warp(self, input, flow,device=torch.device('cpu')):
        B,C,H,W = input.size()

        xx = torch.arange(0,W).view(1,-1).repeat(H,1).to(device)
        yy = torch.arange(0,H).view(-1,1).repeat(1,W).to(device)

        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)

        grid = torch.cat((xx,yy),1).float()

        vgrid = grid + flow
        vgrid[:,0,:,:] = 2*vgrid[:,0,:,:].clone() / max(W-1, 1) - 1.0
        vgrid[:,1,:,:] = 2*vgrid[:,1,:,:].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0,2,3,1)

        output = torch.nn.functional.grid_sample(input,vgrid)
        
        return output
    
    def forward(self, input, state):
        h = state
        inp = torch.cat([input,h],dim=1)
        flows = self.convUV(inp)
        flows = torch.split(flows,2,dim=1)
        warpped_data = []
        for j in range(len(flows)):
            flow = flows[j]
            dd = self.warp(h,-flow,device=self.conv.weight.device)
            warpped_data.append(dd)
        warpped_data = torch.cat(warpped_data,dim=1)
        inpp = torch.cat([input, warpped_data],dim=1)
        comb = self.conv(inpp)
        z,r = torch.split(comb,self.hidden,dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_ = F.leaky_relu(self.rconvI(input)+r*self.rconvH(warpped_data),negative_slope=0.2,inplace=True)
        h = (1-z)*h_ + z*h

        return h