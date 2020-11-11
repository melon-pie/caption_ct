import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, feats_size ,opt):
        super(SelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # 这里的第二维可以算以一个超参数，隐藏层的大小
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)
        self.fc = nn.Linear(opt.images_per_person - opt.neighbor_size+1,opt.images_per_person - opt.neighbor_size+1)
        self.mask = torch.ones(opt.batch_size, 1,opt.images_per_person - opt.neighbor_size+1)
    def forward(self, feats):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        # 先计算自注意力
        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = torch.bmm(weight, value)

        #使用mask矩阵来计算权重信息,权值由全连接层训练参数控制
        self.mask = self.mask.to(device)
        weighted_mask = F.softmax(self.fc(self.mask), dim=-1)
        soft_nsa = torch.bmm(weighted_mask, new_feats).squeeze(1)
        # 输出为(batch_size,2048)维度的特征

        return soft_nsa

class NeighboringSelfAttention(nn.Module):
    def __init__(self, feats_size,opt):
        super(NeighboringSelfAttention, self).__init__()
        self.linear_q = nn.Linear(feats_size, feats_size) # 这里的hid_size有什么影响？即下面的128
        self.linear_k = nn.Linear(feats_size, feats_size)
        self.linear_v = nn.Linear(feats_size, feats_size)
        self.fc = nn.Linear(opt.neighbor_size,opt.neighbor_size)
        self.mask = torch.ones(opt.batch_size, 1, opt.neighbor_size)
    def forward(self, feats):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query = self.linear_q(feats)
        key = self.linear_k(feats)
        value = self.linear_v(feats)

        # 先计算自注意力
        dot = torch.bmm(query, key.transpose(-1, -2)) / math.sqrt(2048)
        weight = F.softmax(dot, dim=-1)
        new_feats = torch.bmm(weight, value)

        # 使用mask矩阵来计算权重信息,权值由全连接层训练参数控制
        self.mask = self.mask.to(device)
        weighted_mask = F.softmax(self.fc(self.mask), dim=-1)
        soft_nsa = torch.bmm(weighted_mask, new_feats)
        # 输出为(batch_size,1,2048)维度的特征
        return soft_nsa

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc_feat_size = opt.fc_feat_size
        self.images_per_person = opt.images_per_person
        self.neighbor_size = opt.neighbor_size

        self.NSA = NeighboringSelfAttention(self.fc_feat_size,opt)
        self.dowsample = nn.Linear(self.images_per_person, self.images_per_person-self.neighbor_size+1)
        self.layernorm = nn.LayerNorm(self.fc_feat_size)
        self.SA  = SelfAttention(self.fc_feat_size,opt)
        # self.fc_class = nn.Linear(self.fc_feat_size, 14)
        # self.layernorm1 = nn.LayerNorm([size, opt.rnn_size]) # 徐骋师兄的LN中的size是什么 ，哪个文件调用 AttentionModel
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
        # 输出为(batch_size,2048)维度的特征
        # new_fc_class = self.fc_class(new_fc_feats)
        return new_fc_feats