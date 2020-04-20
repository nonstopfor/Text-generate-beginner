import torch
import torch.nn as nn
import torch.nn.functional as F


class Lstm(nn.Module):
    # single-layer bidirectional LSTM encoder
    def __init__(self, inp_voc_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()  # python3
        self.embedding = nn.Embedding(inp_voc_dim, emb_dim) # 我们没有预训练过的embedding吗+
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_h = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, input):
        # input:[inp_len, batch_size]
        inp_len = input.size(0)
        embedded = self.dropout(self.embedding(input))
        # embedded:[inp_len, batch_size, emb_dim]
        output, (h_n, c_n) = self.rnn(embedded)
        # output:[inp_len, batch_size, num_dir*enc_hid_dim]
        # h_n,c_n:[num_layers*num_dir,batch_size,enc_hid_dim]
        h_n = torch.relu(self.fc_h(torch.cat((h_n[-2], h_n[-1]), dim=1)))
        c_n = torch.relu(self.fc_c(torch.cat((c_n[-2], c_n[-1]), dim=1)))
        # h_n,,c_n:[batch_size,dec_hid_dim]
        return output, h_n, c_n


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.W = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.randn(dec_hid_dim))

    def forward(self, hidden, enc_outputs):
        # hidden:[batch_size,dec_hid_dim]
        # enc_outputs:[inp_len,batch_size,2*enc_hid_dim]
        inp_len = enc_outputs.size(0)
        hidden = hidden.unsqueeze(0).repeat(inp_len, 1, 1)
        energy = torch.tanh(self.W(torch.cat([hidden, enc_outputs])))
        # energy:[inp_len,batch_size,dec_hid_dim]
        attention = torch.matmul(energy, self.v)
        # attention:[inp_len,batch_size]
        return F.softmax(attention, dim=-1)


class Decoder(nn.Module):
    def __init__(self, out_voc_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.embedding = nn.Embedding(out_voc_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, num_layers=1)
        self.attention = attention(enc_hid_dim, dec_hid_dim)
        self.dec_hid_dim = dec_hid_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, enc_outputs):
        # input = [inputsize, batch_size]
        embedded = self.dropout(self.embedding(input))
        embedded = self.dropout(embedded)
        att_dist = torch.matmul(self.attention(hidden, enc_outputs), embedded)
        # output = [out_voc_dim, batch_size]
        # attention distribution = [attention len, batch_size]
        # out_state = [state size, batch_size]
        output, out_state = self.rnn(embedded, att_dist)
        return output, out_state