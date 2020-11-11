import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLSTMModel(nn.Module):
    def __init__(self, opt):
        super(MultiLSTMModel, self).__init__()
        self.input_size = opt.encoder_lstm_input_size
        self.rnn_size = opt.encoder_lstm_rnn_size
        self.num_layers = opt.encoder_lstm_num_layers
        self.drop_prob_lm = opt.encoder_lstm_drop_prob_lm
        self.img_embed = nn.Linear(opt.fc_feat_size, self.input_size)
        self.core = nn.LSTM(self.input_size, self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm,batch_first=True)
        self.fc_cap = nn.Linear(self.rnn_size, opt.fc_feat_size)
        # self.fc_class = nn.Linear(self.rnn_size,14)
        self.relu = nn.ReLU()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        
    def forward(self, fc_feats):
        batch_size = fc_feats.size(0)
        inputs=self.img_embed(fc_feats)
        state = self.init_hidden(batch_size)
        outputs, state = self.core(inputs, state)
        output_last = outputs[:,-1,:]
        output_last = self.relu(output_last)
        # print(output_last.shape)
        # tags = self.fc(output_last)
        new_fc_feats = self.fc_cap(output_last)
        return new_fc_feats