import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, feats_size ):
        super(SelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # �����hid_size��ʲôӰ�죿�������128
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)

    def forward(self, feats):
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = feats.size(-2)*torch.bmm(weight, value).mean(-2)
        return new_feats

class NeighboringSelfAttention(nn.Module):
    def __init__(self, feats_size):
        super(NeighboringSelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # �����hid_size��ʲôӰ�죿�������128
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)

    def forward(self, feats):
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = torch.bmm(weight, value)
        new_feats = feats.size(-2) * new_feats.mean(-2,keepdim=True)
        return new_feats

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc_feat_size = opt.fc_feat_size
        self.images_per_person = opt.images_per_person
        self.neighbor_size = opt.neighbor_size

        self.NSA = NeighboringSelfAttention(self.fc_feat_size)
        self.dowsample = nn.Linear(self.images_per_person, self.images_per_person-self.neighbor_size+1)
        self.layernorm = nn.LayerNorm(self.fc_feat_size)
        self.SA  = SelfAttention(self.fc_feat_size)
        # self.fc_class = nn.Linear(self.fc_feat_size, 14)
        # self.layernorm1 = nn.LayerNorm([size, opt.rnn_size]) # ���ʦ�ֵ�LN�е�size��ʲô ���ĸ��ļ����� AttentionModel
        # self.layernorm2 = nn.LayerNorm([size, opt.rnn_size])
    def forward(self, fc_feats):
        batch_size = fc_feats.size(0)
        nsa_feats = fc_feats.data.new( batch_size,1, 2048).uniform_(0, 1)
        for i in range(self.images_per_person):
            if i<self.neighbor_size-1 :
                continue
            elif i == self.neighbor_size-1:
                tem = self.NSA(fc_feats[:, :i+1,:])
                nsa_feats = tem
            else:
                tem = self.NSA(fc_feats[:, i+1-self.neighbor_size:i+1,:])
                nsa_feats = torch.cat([nsa_feats, tem],1)
        new_fc_feats = self.dowsample(fc_feats.transpose(-1, -2)).transpose(-1, -2)
        new_fc_feats = self.layernorm(new_fc_feats+nsa_feats)
        new_fc_feats = self.SA(new_fc_feats)
        # new_fc_class = self.fc_class(new_fc_feats)
        return new_fc_feats