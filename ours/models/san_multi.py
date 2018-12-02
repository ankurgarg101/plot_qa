"""
Model Definition for Stacked Attention Network.
Source Code Reference: https://github.com/rshivansh/San-Pytorch/blob/master/misc/san.py
Paper Reference: https://arxiv.org/pdf/1511.02274.pdf

Changed to fetch attention from text and images(bboxes too, depends on input)
"""

import torch
import torch.nn as nn
import math
class SAN_MULTI(nn.Module):
	def __init__(self, input_img_size, input_ques_size, att_size, output_size, use_gpu, input_text_size=None, img_feat_size=None, n_heads = 1,dot_product_att = False):
		
		super(SAN_MULTI, self).__init__()
		self.use_gpu = use_gpu
		self.input_img_size = input_img_size
		self.input_ques_size = input_ques_size
		self.input_text_size = input_text_size
		self.att_size = att_size
		self.output_size = output_size
		self.dot_product_att = dot_product_att
		if input_text_size is None:
			self.use_text = False
		else:
			self.use_text = True

		if img_feat_size is None:
			self.use_global_img = False
		else:
			self.use_global_img = True
			self.img_feat_size = img_feat_size

		self.n_heads = n_heads

		print('Using Global Img Feats', self.use_global_img)

		# Specify the non-linearities
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=-1)

		self.all_comb_size = 2
		# Stack 1 layers
		self.W_qa_img_1 = nn.Linear(input_ques_size, self.n_heads*att_size)
		self.W_img_ques_emb = nn.Linear(input_img_size, input_ques_size)
		self.W_ia_1 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
		if not dot_product_att:
			self.W_p_img_1 = nn.Linear(att_size, 1)
		
		if self.use_global_img:
			self.W_qa_global_img_1 = nn.Linear(input_ques_size, self.n_heads*att_size)
			self.W_global_img_ques_emb = nn.Linear(img_feat_size, input_ques_size)
			self.W_global_ia_1 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
			if not dot_product_att:
				self.W_p_global_img_1 = nn.Linear(att_size, 1)

			self.all_comb_size +=  1

		if self.input_text_size is not None:
			self.W_qa_text_1 = nn.Linear(input_ques_size, self.n_heads*att_size)
			self.W_txt_ques_emb = nn.Linear(input_text_size, input_ques_size)
			self.W_ta_1 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
			if not dot_product_att:
				self.W_p_text_1 = nn.Linear(att_size, 1)

			self.all_comb_size += 1
			
		# Stack 2 layers
		self.W_qa_img_2 = nn.Linear(input_ques_size, self.n_heads*att_size)
		self.W_ia_2 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
		if not dot_product_att:
			self.W_p_img_2 = nn.Linear(att_size, 1)

		if self.use_global_img:
			self.W_qa_global_img_2 = nn.Linear(input_ques_size, self.n_heads*att_size)
			self.W_global_ia_2 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
			if not dot_product_att:
				self.W_p_global_img_2 = nn.Linear(att_size, 1)

		if self.input_text_size is not None:
			self.W_qa_text_2 = nn.Linear(input_ques_size, self.n_heads*att_size)
			self.W_ta_2 = nn.Linear(input_ques_size, self.n_heads*att_size, bias=False)
			if not dot_product_att:
				self.W_p_text_2 = nn.Linear(att_size, 1)
			
		self.img_comb = nn.Linear(self.n_heads*input_ques_size, input_ques_size)
		self.txt_comb = nn.Linear(self.n_heads*input_ques_size, input_ques_size)
		self.global_comb = nn.Linear(self.n_heads*input_ques_size, input_ques_size)

		#self.all_comb = nn.Linear(self.all_comb_size,input_ques_size)
		# Final fc layer
		self.W_u = nn.Linear(input_ques_size, output_size)

	def forward(self, ques_feats, img_feats, num_boxes = None, text_feats = None, num_texts = None, global_img_feats=None):  # ques_feats -- [batch, d] | img_feats -- [batch_size, img_seq_size/max_num_bars, input_img_size] | text_feats -- [batch_size,max_num_text,input_text_size]
		batch_size = ques_feats.size(0)

		if num_boxes is not None:
			max_num_boxes = img_feats.size(1)
			img_mask = torch.Tensor(batch_size,max_num_boxes)
			inv_img_mask = torch.Tensor(batch_size,self.n_heads,max_num_boxes)
			inv_img_mask = inv_img_mask.type(torch.ByteTensor)
			if self.use_gpu:
				img_mask = img_mask.cuda()
				inv_img_mask = inv_img_mask.cuda()
			# for b in range(max_num_boxes):
			# 	img_mask[:,b] = (b < num_boxes)
			for b in range(max_num_boxes):
				for h in range(self.n_heads):
					inv_img_mask[:, h, b] = (b >= num_boxes)

		if self.use_text:
			max_num_texts = text_feats.size(1)
			text_mask = torch.Tensor(batch_size,max_num_texts)
			inv_text_mask = torch.Tensor(batch_size,self.n_heads,max_num_texts)
			inv_text_mask = inv_text_mask.type(torch.ByteTensor)			
			if self.use_gpu:
				text_mask = text_mask.cuda()
				inv_text_mask = inv_text_mask.cuda()
			# for b in range(max_num_texts):
			# 	text_mask[:,b] = (b < num_texts)
			for b in range(max_num_texts):
				for h in range(self.n_heads):
					inv_text_mask[:, h, b] = (b >= num_texts)

		# Stack 1
		ques_emb_img_1 = self.W_qa_img_1(ques_feats).view(batch_size,1,self.n_heads,self.att_size).repeat(1,img_feats.size(1),1,1).view(-1,self.att_size)
		img_ques_emb = self.tanh(self.W_img_ques_emb(img_feats)) #computed only once
		img_emb_1 = self.W_ia_1(img_ques_emb).view(batch_size,img_ques_emb.size(1),self.n_heads,self.att_size).view(-1,self.att_size)

		if self.dot_product_att:
			h1_emb = torch.bmm(ques_emb_img_1.unsqueeze(1),img_emb_1.unsqueeze(2)).view(batch_size,img_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
		else:
			h1_emb = self.W_p_img_1(self.tanh(ques_emb_img_1+img_emb_1)).view(batch_size,img_ques_emb.size(1),self.n_heads).permute(0,2,1)

		if num_boxes is not None:
			h1_emb.data.masked_fill_(inv_img_mask, -float("inf"))
		
		p1 = self.softmax(h1_emb)

		# Weighted sum
		p1 = p1.view(batch_size,self.n_heads,-1)
		img_att1 = torch.bmm(p1, img_ques_emb).view(batch_size,-1)
		img_att1 = self.tanh(self.img_comb(img_att1))
		comb_att1 = ques_feats + img_att1
		#comb_att1 = torch.cat((img_att1,ques_feats),dim=1)
		if self.use_text:
			ques_emb_text_1 = self.W_qa_text_1(ques_feats).view(batch_size,1,self.n_heads,self.att_size).repeat(1,text_feats.size(1),1,1).view(-1,self.att_size)
			text_ques_emb = self.tanh(self.W_txt_ques_emb(text_feats)) 
			text_emb_1 = self.W_ta_1(text_ques_emb).view(batch_size,text_ques_emb.size(1),self.n_heads,self.att_size).view(-1,self.att_size)
			
			if self.dot_product_att:
				h1_emb = torch.bmm(ques_emb_text_1.unsqueeze(1),text_emb_1.unsqueeze(2)).view(batch_size,text_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
			else:
				h1_emb = self.W_p_text_1(self.tanh(ques_emb_text_1+text_emb_1)).view(batch_size,text_ques_emb.size(1),self.n_heads).permute(0,2,1)

			h1_emb.data.masked_fill_(inv_text_mask, -float("inf"))

			p1 = self.softmax(h1_emb)
			p1 = p1.view(batch_size,self.n_heads,-1)
			# p1 = p1_w * text_mask
			text_att1 = torch.bmm(p1, text_ques_emb).view(batch_size,-1)
			text_att1 = self.tanh(self.txt_comb(text_att1))

			comb_att1 += text_att1
			#comb_att1 = torch.cat((img_att1,comb_att1),dim=1)

		if self.use_global_img:
			ques_emb_global_img_1 = self.W_qa_global_img_1(ques_feats).view(batch_size,1,self.n_heads,self.att_size).repeat(1,global_img_feats.size(1),1,1).view(-1,self.att_size)
			global_img_ques_emb = self.tanh(self.W_global_img_ques_emb(global_img_feats))
			global_img_emb_1 = self.W_global_ia_1(global_img_ques_emb).view(-1,self.att_size)

			if self.dot_product_att:
				h1_emb = torch.bmm(ques_emb_global_img_1.unsqueeze(1),global_img_emb_1.unsqueeze(2)).view(batch_size,global_img_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
			else:
				h1_emb = self.W_p_global_img_1(self.tanh(ques_emb_global_img_1+global_img_emb_1)).view(batch_size,global_img_ques_emb.size(1),self.n_heads).permute(0,2,1)

			p1 = self.softmax(h1_emb)

			p1 = p1.view(batch_size,self.n_heads,-1)
			
			global_img_att1 = torch.bmm(p1, global_img_ques_emb).view(batch_size,-1)
			global_img_att1 = self.tanh(self.global_comb(global_img_att1))
			
			comb_att1 += global_img_att1
			#comb_att1 = torch.cat((global_img_att1,comb_att1),dim=1)

		u1 = comb_att1
		#u1 = ques_feats + comb_att1

		# Stack 2
		ques_emb_img_2 = self.W_qa_img_2(u1).view(batch_size,1,self.n_heads,self.att_size).repeat(1,img_feats.size(1),1,1).view(-1,self.att_size)
		img_emb_2 = self.W_ia_2(img_ques_emb).view(batch_size,img_ques_emb.size(1),self.n_heads,self.att_size).view(-1,self.att_size)

		if self.dot_product_att:
			h2_emb = torch.bmm(ques_emb_img_2.unsqueeze(1),img_emb_2.unsqueeze(2)).view(batch_size,img_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
		else:
			h2_emb = self.W_p_img_2(self.tanh(ques_emb_img_2+img_emb_2)).view(batch_size,img_ques_emb.size(1),self.n_heads).permute(0,2,1)

		if num_boxes is not None:
			h2_emb.data.masked_fill_(inv_img_mask, -float("inf"))

		p2 = self.softmax(h2_emb)

		p2 = p2.view(batch_size,self.n_heads,-1)

		# Weighted sum
		img_att2 = torch.bmm(p2, img_ques_emb).view(batch_size,-1)
		img_att2 = self.tanh(self.img_comb(img_att2))
		
		comb_att2 = u1 + img_att2
		#comb_att2 = torch.cat((img_att2,u1),dim=1)

		if self.use_text:
			ques_emb_text_2 = self.W_qa_text_2(u1).view(batch_size,1,self.n_heads,self.att_size).repeat(1,text_feats.size(1),1,1).view(-1,self.att_size)
			text_emb_2 = self.W_ta_2(text_ques_emb).view(batch_size,text_ques_emb.size(1),self.n_heads,self.att_size).view(-1,self.att_size) 
			
			if self.dot_product_att:
				h2_emb = torch.bmm(ques_emb_text_2.unsqueeze(1),text_emb_2.unsqueeze(2)).view(batch_size,text_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
			else:
				h2_emb = self.W_p_text_2(self.tanh(ques_emb_text_2+text_emb_2)).view(batch_size,text_ques_emb.size(1),self.n_heads).permute(0,2,1)

			h2_emb.data.masked_fill_(inv_text_mask, -float("inf"))

			p2 = self.softmax(h2_emb)
			p2 = p2.view(batch_size,self.n_heads,-1)
			# p2 = p2_w * text_mask
			text_att2 = torch.bmm(p2, text_ques_emb).view(batch_size,-1)
			text_att2 = self.tanh(self.txt_comb(text_att2))

			comb_att2 += text_att2
			#comb_att2 = torch.cat((text_att2,comb_att2),dim=1)

		if self.use_global_img:
			ques_emb_global_img_2 = self.W_qa_global_img_2(u1).view(batch_size,1,self.n_heads,self.att_size).repeat(1,global_img_feats.size(1),1,1).view(-1,self.att_size)
			global_img_emb_2 = self.W_global_ia_2(global_img_ques_emb).view(batch_size,global_img_ques_emb.size(1),self.n_heads,self.att_size).view(-1,self.att_size) 
			
			if self.dot_product_att:
				h2_emb = torch.bmm(ques_emb_global_img_2.unsqueeze(1),global_img_emb_2.unsqueeze(2)).view(batch_size,global_img_ques_emb.size(1),self.n_heads).permute(0,2,1)/math.sqrt(self.att_size)
			else:
				h2_emb = self.W_p_global_img_2(self.tanh(ques_emb_global_img_2+global_img_emb_2)).view(batch_size,global_img_ques_emb.size(1),self.n_heads).permute(0,2,1)
			
			p2 = self.softmax(h2_emb)
			
			global_img_att2 = torch.bmm(p2, global_img_ques_emb).view(batch_size,-1)
			global_img_att2 = self.tanh(self.global_comb(global_img_att2))
			
			comb_att2 += global_img_att2
			#comb_att2 = torch.cat((global_img_att2,comb_att2),dim=1)

		u2 = comb_att2
		#u2 = self.tanh(self.all_comb(comb_att2))

		# Final softmax outputs
		scores = self.W_u(u2)

		return scores