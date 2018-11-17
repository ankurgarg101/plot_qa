import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, rnn_size, num_layers, lstm_dropout, seq_length, use_gpu):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()

        self.use_gpu = use_gpu
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.E = nn.Embedding(vocab_size, emb_size)
        # self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=True,
                batch_first=True, dropout=lstm_dropout,bidirectional=True)

    def forward(self, ques_vectors, ques_lengths):            # forward(self, ques_vec, ques_len) | ques_vec: [batch_size, 26]
        # batch_size, pad_len = ques_vectors.size()

        # Add 1 to vocab_size, since word idx from 0 to vocab_size inclusive
        # one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2, ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)

        # To remove additional column in one_hot, use slicing
        # one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
        # if self.use_gpu and torch.cuda.is_available():
        #     one_hot_vec = one_hot_vec.cuda()

        # x = self.lookuptable(one_hot_vec)

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        # emb_vec = self.dropout(self.tanh(x))


        # # h: [batch_size or B, 26 or W, hidden_size]
        # h, _ = self.LSTM(emb_vec)

        # x = torch.LongTensor(ques_len - 1)
        # mask = torch.zeros(B, W).scatter_(1, x.view(-1, 1), 1)
        # mask = Variable(mask.view(B, W, 1), requires_grad=False)
        # if self.use_gpu and torch.cuda.is_available():
        #     mask = mask.cuda()

        # h = h.transpose(1,2)
        # # print(h.size(), mask.size())

        # output: [B, hidden_size]
        # return torch.bmm(h, mask).view(B, -1)

        # Embedding Layer
        ques_embedded = self.E(ques_vectors)

        # LSTM Layer
        packed_input = nn.utils.rnn.pack_padded_sequence(ques_embedded, ques_lengths, batch_first=True)
        ques_output, _ = self.lstm(packed_input)
        unpacked_output = nn.utils.rnn.pad_packed_sequence(ques_output, batch_first=True)[0]

        batch_size, padded_len, hidden_size = unpacked_output.size()

        # Choose the output at ques_length-1 index for all examples in the batch
        x = torch.LongTensor(ques_lengths - 1).unsqueeze(1)
        mask = torch.zeros(batch_size, padded_len).scatter_(1, x, 1).unsqueeze(1)
        final_output = torch.bmm(mask, unpacked_output).squeeze(1)

        return final_output
