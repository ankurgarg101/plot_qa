"""
Model Definition for Stacked Attention Network.
Source Code Reference: https://github.com/rshivansh/San-Pytorch/blob/master/misc/san.py
Paper Reference: https://arxiv.org/pdf/1511.02274.pdf
"""

import torch
import torch.nn as nn

class SAN(nn.Module):
    def __init__(self, input_size, att_size, img_seq_size, output_size):
        
        super(SAN, self).__init__()
        
        # d = input_size | m = img_seq_size | k = att_size
        self.input_size = input_size
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size

        # Specify the non-linearities
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Stack 1 layers
        self.W_qa_1 = nn.Linear(input_size, att_size)
        self.W_ia_1 = nn.Linear(input_size, att_size, bias=False)
        self.W_p_1 = nn.Linear(att_size, 1)

        # Stack 2 layers
        self.W_qa_2 = nn.Linear(input_size, att_size)
        self.W_ia_2 = nn.Linear(input_size, att_size, bias=False)
        self.W_p_2 = nn.Linear(att_size, 1)

        # Final fc layer
        self.W_u = nn.Linear(input_size, output_size)

    def forward(self, ques_feats, img_feats):  # ques_feats -- [batch, d] | img_feats -- [batch_size, 14, 14, d]
        batch_size = ques_feats.size(0)

        img_feats = img_feats.view(batch_size, self.img_seq_size, -1)

        # Stack 1
        ques_emb_1 = self.W_qa_1(ques_feats) 
        img_emb_1 = self.W_ia_1(img_feats)

        h1_emb = self.W_p_1(img_emb_1 + ques_emb_1.unsqueeze(1)).squeeze(2)
        p1 = self.softmax(h1_emb)

        # Weighted sum
        img_att1 = torch.bmm(p1.unsqueeze(1), img_feats).squeeze(1)
        u1 = ques_feats + img_att1

        # Stack 2
        ques_emb_2 = self.W_qa_2(u1)
        img_emb_2 = self.W_ia_2(img_feats)

        h2_emb = self.W_p_2(img_emb_2 + ques_emb_2.unsqueeze(1)).squeeze(2)
        p2 = self.softmax(h2_emb)

        # Weighted sum
        img_att2 = torch.bmm(p2.unsqueeze(1), img_feats).squeeze(1)
        u2 = u1 + img_att2

        # Final softmax outputs
        scores = self.softmax(self.W_u(u2))

        return scores