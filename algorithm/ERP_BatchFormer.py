
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from algorithm.FAN import FANLayer
from algorithm.util import *
# from algorithm.Module import *






class ERP_BatchFormer(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes,
                hid_dim,
                base_model,
                dropout_rate=0.1, 
                ):
        
        super().__init__()


      
        self.base_model=deepcopy(base_model)


        encoder_layer=TransformerEncoderLayer(d_model=n_channels,nhead=1,dim_feedforward=hid_dim,dropout=dropout_rate)
        self.regressor=TransformerEncoder(encoder_layer,num_layers=4)


        self.w=nn.Parameter(torch.rand(2),requires_grad=True)


    def forward(self,x):
        n_batch, n_channel, n_sample = x.shape

        x_t=x.transpose(1,2)

        template=self.regressor(x_t).transpose(1,2)
        

        out=self.base_model(template*torch.relu(self.w[0])+x*torch.relu(self.w[1]))


        return out,template*torch.relu(self.w[0])
    

'''
class MultiTaskFan_mod(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes,
                hid_dim1,
                hid_dim2,
                base_model,
                dropout_rate=0., 
                ):
        
        super().__init__()

        
      
        self.base_model=deepcopy(base_model)




        # self.regressor=nn.LSTM(n_channels,
        #                        hid_dim2,
        #                        num_layers=4,
        #                        bidirectional=True,
        #                        batch_first=True,
        #                        dropout=dropout_rate)
        # self.reg_fc=nn.Linear(hid_dim2,n_channels,bias=False)

        # self.gate=nn.Linear(n_channels*n_samples,1)
        # self.pos_encoder = PositionalEncoding(d_model=n_channels, dropout=dropout_rate)
        encoder_layer=TransformerEncoderLayer(d_model=n_channels,nhead=1,dim_feedforward=hid_dim2)
        self.regressor=TransformerEncoder(encoder_layer,num_layers=4)

        self.reg_fc=nn.Linear(hid_dim1,n_channels,bias=False)

        # self.w=nn.Parameter(torch.tensor(0.1))
        self.w=nn.Parameter(torch.rand(2),requires_grad=True)

        # self.fusion_fc = nn.Linear(n_channels * 2, n_channels, bias=False)

    def forward(self,x):
        n_batch, n_channel, n_sample = x.shape

        x_t=x.transpose(1,2)
        # x_fan=self.FAN1(x_t)
        # x_fan_o=self.FAN2(x_fan)


        # template=self.regressor(x_fan_o*torch.sigmoid(self.w[0])+x_t*torch.sigmoid(self.w[1]))

        # template,_=self.regressor(x_fan_o*torch.sigmoid(self.w)+x_t)
        template=self.regressor(x_t)
        # template=self.reg_fc(template)

        
        template=template.transpose(1,2)

        # w=self.gate(x.flatten(start_dim=1))
        # w=F.leaky_relu(self.w).unsqueeze(1)


        out=self.base_model(template+x)

        # out=self.base_model(template*torch.sigmoid(self.w2[0])+x*torch.sigmoid(self.w2[1]))

        # p_encoder_feature = self.pos_encoder(template)
        # template=self.regressor(p_encoder_feature)


        return out,template
    



class MultiTaskFan_mod2(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes,
                hid_dim1,
                hid_dim2,
                base_model,
                dropout_rate=0., 
                ):
        
        super().__init__()

        
      
        self.base_model=deepcopy(base_model)



        encoder_layer=TransformerEncoderLayer(d_model=n_channels,nhead=1,dim_feedforward=hid_dim2)
        self.regressor=TransformerEncoder(encoder_layer,num_layers=4)

        self.reg_fc=nn.Linear(hid_dim1,n_channels,bias=False)


        self.w=nn.Parameter(torch.rand(2),requires_grad=True)


    def forward(self,x):
        n_batch, n_channel, n_sample = x.shape

        x_t=x.transpose(1,2)
        template=self.regressor(x_t).transpose(1,2)
        

        out=self.base_model(template*torch.relu(self.w[0])+x*torch.relu(self.w[1]))


        return out,template*torch.relu(self.w[0])
    

'''