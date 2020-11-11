import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .CapEncoderLSTM import MultiLSTMModel


class SelfAttention(nn.Module):
    def __init__(self, feats_size,opt):
        super(SelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # 这里的hid_size有什么影响？即下面的128
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)
        self.lstm = MultiLSTMModel(opt)
    def forward(self, feats):
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = torch.bmm(weight, value)
        new_feats = self.lstm(new_feats)
        return new_feats

class NeighboringSelfAttention(nn.Module):
    def __init__(self, feats_size,opt):
        super(NeighboringSelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # 这里的hid_size有什么影响？即下面的128
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)
        self.lstm = MultiLSTMModel(opt)
    def forward(self, feats):
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = torch.bmm(weight, value)
        new_feats = self.lstm(new_feats).unsqueeze(1)
        return new_feats


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc_feat_size = opt.fc_feat_size
        self.images_per_person = opt.images_per_person
        self.neighbor_size = opt.neighbor_size

        self.NSA = NeighboringSelfAttention(self.fc_feat_size, opt)
        # self.dowsample = nn.Linear(self.images_per_person, self.images_per_person-self.neighbor_size+1)
        self.layernorm = nn.LayerNorm(self.fc_feat_size)

        # self.NSA01 = NeighboringSelfAttention(self.fc_feat_size, opt)
        # self.dowsample1 = nn.Linear(self.images_per_person-self.neighbor_size+1, self.images_per_person-2*self.neighbor_size+2)
        # self.layernorm1 = nn.LayerNorm(self.fc_feat_size)

        self.SA = SelfAttention(self.fc_feat_size, opt)

    def forward(self, fc_feats):
        batch_size = fc_feats.size(0)
        # 第一层
        nsa_feats = fc_feats.data.new( batch_size,1, 2048).uniform_(0, 1)
        for i in range(self.images_per_person):
            if i<self.neighbor_size-1 :
                continue
            elif i == self.neighbor_size-1:
                tem = self.NSA(fc_feats[:, :i+1, :])
                nsa_feats = tem
            else:
                tem = self.NSA(fc_feats[:, i+1-self.neighbor_size:i+1, :])
                nsa_feats = torch.cat([nsa_feats, tem], 1)
        # new_fc_feats = self.dowsample(fc_feats.transpose(-1, -2)).transpose(-1, -2)
        # new_nsa_feats = self.layernorm(new_fc_feats+nsa_feats)
        new_nsa_feats = self.layernorm(nsa_feats)

        # 第二层
        # nsa01_feats = new_nsa_feats.data.new(batch_size, 1, 2048).uniform_(0, 1)
        # for i in range(self.images_per_person-self.neighbor_size+1):
        #     if i < self.neighbor_size-1:
        #         continue
        #     elif i == self.neighbor_size-1:
        #         tem = self.NSA01(nsa01_feats[:, :i+1, :])
        #         nsa01_feats = tem
        #     else:
        #         tem = self.NSA01(new_nsa_feats[:, i+1-self.neighbor_size:i+1, :])
        #         nsa01_feats = torch.cat([nsa01_feats, tem], 1)
        # new_nsa_feats_ = self.dowsample1(nsa_feats.transpose(-1, -2)).transpose(-1, -2)
        # new_nsa01_feats = self.layernorm1(new_nsa_feats_ + nsa01_feats)

        sa_feats = self.SA(new_nsa_feats)
        return sa_feats, new_nsa_feats


