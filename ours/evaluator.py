"""
Module that evaluates a specified checkpoint against the val_easy and val_hard datasets
"""

import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torchvision
import h5py

def load_models(models, params, load_model_dir):

	if 'ques_model' in models:
		models['ques_model'].load_state_dict(torch.load(
			os.path.join(load_model_dir, 'question_model.pkl')))
	
	if 'att_model' in models:
		models['att_model'].load_state_dict(torch.load(
			os.path.join(load_model_dir, 'attention_model.pkl')))

	if 'img_model' in models:
		models['img_model'].load_state_dict(torch.load(
			os.path.join(load_model_dir, 'image_model.pkl')))

	if 'roi_model' in models:
		models['roi_model'].load_state_dict(torch.load(
		os.path.join(load_model_dir, 'roi_model.pkl')))

	if 'text_model' in models:
		models['text_model'].load_state_dict(torch.load(
			os.path.join(load_model_dir, 'text_model.pkl')))

	return models

def eval_model(models, dataset, params, extra_params):
	
	# Construct Data loader
	dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)
	
	if params['use_gpu'] and torch.cuda.is_available():
		print('Initialized Cuda Models')
		
		for mname in models:
			models[mname] = models[mname].cuda()

	print('Loading Models')
	
	load_model_dir = os.path.join(params['checkpoint_path'])
	
	print('Loading model files from folder: %s' % load_model_dir)
	
	models = load_models(models, params, load_model_dir)

	if params['use_roi'] or params['load_roi']:
		roi_save_file = h5py.File(params['roi_save_file'])

	accuracies = []
	# Call train() on all models for training
	for m in models:
		models[m].eval()
		
	if params['use_gpu'] and torch.cuda.is_available():
		pred = lambda x: np.argmax(x.cpu().detach().numpy(), axis=1)
	else:    
		pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

	running_val_loss = 0.0
	for i, batch in enumerate(dataloader):

		images = batch['image']
		questions = batch['ques']
		ques_lens = batch['ques_len']
		answers = batch['ans']
		bar_lens = batch['bar_len']
		text_lens = batch['text_len']
		bar_bboxes = batch['bar_bboxes']
		text_bboxes = batch['text_bboxes']
		text_vals = batch['text_vals']
		text_types = batch['text_types']
		ids = batch['id']
		if params['load_roi']:
			roi_feats = batch['roi_feats']

		# Sort the examples in reverse order of sentence length
		_, sort_idxes = torch.sort(ques_lens, descending=True)
		images = images[sort_idxes, :, :, :]
		questions = questions[sort_idxes, :]
		ques_lens = ques_lens[sort_idxes]
		answers = answers[sort_idxes, :]
		answers = answers.squeeze(1)
		bar_lens = bar_lens[sort_idxes]
		text_lens = text_lens[sort_idxes]
		bar_bboxes = bar_bboxes[sort_idxes]
		text_bboxes = text_bboxes[sort_idxes]
		text_vals = text_vals[sort_idxes]
		text_types = text_types[sort_idxes]
		ids = [ids[i] for i in sort_idxes]
		if params['load_roi']:
			roi_feats = roi_feats[sort_idxes]
		
		# print (images)
		# print (questions)
		# print (ques_lens)
		# print (answers)

		if (params['use_gpu'] and torch.cuda.is_available()):
			images = images.cuda()
			questions = questions.cuda()
			answers = answers.cuda()
			bar_lens = bar_lens.cuda()
			text_lens = text_lens.cuda()
			bar_bboxes = bar_bboxes.cuda()
			text_bboxes = text_bboxes.cuda()
			text_vals = text_vals.cuda()
			text_types = text_types.cuda()
			ids = ids
			if params['load_roi']:
				roi_feats = roi_feats.cuda()

		ques_emb = models['ques_model'].forward(questions, ques_lens)
		
		global_img_feats = None
		if params['load_roi']:
			img_emb = roi_feats

			if params['use_global_img']:
				assert 'img_model' in models
				img_emb = models['img_model'].forward(images)
				global_img_feats = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

		else:
			img_emb = models['img_model'].forward(images)

			if params['use_global_img']:
				global_img_feats = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

			if params['use_roi']:

				box_idx = torch.as_tensor(np.repeat(range(len(images)), extra_params['max_num_bars']), dtype=torch.int)
				if (params['use_gpu'] and torch.cuda.is_available()):
					box_idx = box_idx.cuda()

				img_emb = models['roi_model'].forward(img_emb, bar_bboxes, box_idx )

				
				if params['roi_save_file']:
					for b in range(len(ids)):
						curr_id = ids[b]
						curr_emb = img_emb[b,:].cpu().numpy()
						if curr_id not in list(roi_save_file.keys()):
							roi_save_file.create_dataset(curr_id,data=curr_emb)

			else:
				img_emb = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

		if params['use_text']:
			text_emb = models['text_model'].forward(text_vals)
			text_emb = torch.cat((text_emb, text_types), dim=2)
		else:
			text_emb = None

		if params['use_pos']:

			if params['use_text']:
				text_emb = torch.cat((text_emb, text_bboxes), dim=2)

			if params['use_roi']:
				img_emb = torch.cat((img_emb, bar_bboxes), dim=2)
		
		if params['use_roi'] or params['load_roi']:
			output = models['att_model'].forward(ques_emb, img_emb, num_boxes=bar_lens, text_feats=text_emb, num_texts=text_lens, global_img_feats=global_img_feats)
		else:
			output = models['att_model'].forward(ques_emb, img_emb, num_boxes=None, text_feats=text_emb, num_texts=text_lens, global_img_feats=global_img_feats)

		if (params['use_gpu'] and torch.cuda.is_available()):
			answers = answers.cpu()

		output_preds = pred(output)
		accuracies.extend( output_preds ==  answers.detach().numpy())
		print('Interim Val Accurracy: %.4f' % (np.mean(accuracies)))

	with open(os.path.join(params['checkpoint_path'], '{}_acc.json'.format(params['split'])), 'w') as f:
		print('Val Accurracy: %.4f' % (np.mean(accuracies)))        