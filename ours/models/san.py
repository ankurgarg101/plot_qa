"""
Model Definition for Stacked Attention Network.
Source Code Reference: https://github.com/rshivansh/San-Pytorch/blob/master/misc/san.py
Paper Reference: https://arxiv.org/pdf/1511.02274.pdf

Changed to fetch attention from text and images(bboxes too, depends on input)
"""

import torch
import torch.nn as nn

class SAN(nn.Module):
	def __init__(self, input_img_size, input_ques_size, att_size, output_size, use_gpu, input_text_size=None, ):
		
		super(SAN, self).__init__()
		self.use_gpu = use_gpu
		self.input_img_size = input_img_size
		self.input_ques_size = input_ques_size
		self.input_text_size = input_text_size
		self.att_size = att_size
		self.output_size = output_size
		if input_text_size is None:
			self.use_text = False
		else:
			self.use_text = True

		# Specify the non-linearities
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)

		# Stack 1 layers
		self.W_qa_img_1 = nn.Linear(input_ques_size, att_size)
		self.W_ia_1 = nn.Linear(input_img_size, att_size, bias=False)
		self.W_p_img_1 = nn.Linear(att_size, 1)
		self.W_comb_1 = nn.Linear(self.input_img_size,self.input_ques_size)

		if self.input_text_size is not None:
			self.W_qa_text_1 = nn.Linear(input_ques_size, att_size)
			self.W_ta_1 = nn.Linear(input_text_size, att_size, bias=False)
			self.W_p_text_1 = nn.Linear(att_size, 1)
			#combine attentions
			self.W_comb_1 = nn.Linear(self.input_img_size+self.input_text_size,self.input_ques_size)
		
		# Stack 2 layers
		self.W_qa_img_2 = nn.Linear(input_ques_size, att_size)
		self.W_ia_2 = nn.Linear(input_img_size, att_size, bias=False)
		self.W_p_img_2 = nn.Linear(att_size, 1)
		self.W_comb_2 = nn.Linear(self.input_img_size,self.input_ques_size)

		if self.input_text_size is not None:
			self.W_qa_text_2 = nn.Linear(input_ques_size, att_size)
			self.W_ta_2 = nn.Linear(input_text_size, att_size, bias=False)
			self.W_p_text_2 = nn.Linear(att_size, 1)
			#combine attentions
			self.W_comb_2 = nn.Linear(self.input_img_size+self.input_text_size,self.input_ques_size)

		# Final fc layer
		self.W_u = nn.Linear(input_ques_size, output_size)

	def forward(self, ques_feats, img_feats, num_boxes = None, text_feats = None, num_texts = None):  # ques_feats -- [batch, d] | img_feats -- [batch_size, img_seq_size/max_num_bars, input_img_size] | text_feats -- [batch_size,max_num_text,input_text_size]
		batch_size = ques_feats.size(0)

		if num_boxes is not None:
			max_num_boxes = img_feats.size(1)
			img_mask = torch.Tensor(batch_size,max_num_boxes)
			if self.use_gpu:
				img_mask = img_mask.cuda()
			for b in range(max_num_boxes):
				img_mask[:,b] = (b < num_boxes)

		if num_texts is not None:
			max_num_texts = text_feats.size(1)
			text_mask = torch.Tensor(batch_size,max_num_texts)
			if self.use_gpu:
				text_mask = text_mask.cuda()
			for b in range(max_num_texts):
				text_mask[:,b] = (b < num_texts)

		# Stack 1
		ques_emb_img_1 = self.W_qa_img_1(ques_feats) 
		img_emb_1 = self.W_ia_1(img_feats)

		h1_emb = self.W_p_img_1(img_emb_1 + ques_emb_img_1.unsqueeze(1)).squeeze(2)
		if num_boxes is not None:
			h1_emb *= img_mask
		p1 = self.softmax(h1_emb)

		# Weighted sum
		img_att1 = torch.bmm(p1.unsqueeze(1), img_feats).squeeze(1)
		
		if self.use_text:
			ques_emb_text_1 = self.W_qa_text_1(ques_feats) 
			text_emb_1 = self.W_ta_1(text_feats)
			h1_emb = self.W_p_text_1(text_emb_1 + ques_emb_text_1.unsqueeze(1)).squeeze(2)
			if num_texts is not None:
				h1_emb *= text_mask
			p1 = self.softmax(h1_emb)
			text_att1 = torch.bmm(p1.unsqueeze(1), text_feats).squeeze(1)
			comb_att1 = self.W_comb_1(torch.cat((img_att1,text_att1),dim=1))
		else:
			comb_att1 = self.W_comb_1(img_att1)

		u1 = ques_feats + comb_att1

		# Stack 2
		ques_emb_img_2 = self.W_qa_img_2(u1)
		img_emb_2 = self.W_ia_2(img_feats)

		h2_emb = self.W_p_img_2(img_emb_2 + ques_emb_img_2.unsqueeze(1)).squeeze(2)
		if num_boxes is not None:
			h2_emb *= img_mask
		p2 = self.softmax(h2_emb)

		# Weighted sum
		img_att2 = torch.bmm(p2.unsqueeze(1), img_feats).squeeze(1)

		if self.use_text:
			ques_emb_text_2 = self.W_qa_text_2(u1)
			text_emb_2 = self.W_ta_2(text_feats) 
			h2_emb = self.W_p_text_2(text_emb_2 + ques_emb_text_2.unsqueeze(1)).squeeze(2)
			if num_texts is not None:
				h2_emb *= text_mask
			p2 = self.softmax(h2_emb)
			text_att2 = torch.bmm(p2.unsqueeze(1), text_feats).squeeze(1)
			comb_att2 = self.W_comb_1(torch.cat((img_att2,text_att2),dim=1))
		else:
			comb_att2 = self.W_comb_2(img_att2)

		u2 = u1 + comb_att2

		# Final softmax outputs
		scores = self.softmax(self.W_u(u2))

		return scores