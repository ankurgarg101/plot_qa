"""
Module that trains the SAN network
TODO: Make sure it automatically resumes from a previous checkpoint.
"""

import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torchvision
import h5py

def adjust_learning_rate(optimizer, epoch, lr, learning_rate_decay_every):
	
	# Sets the learning rate to the initial LR decayed by 10 every learning_rate_decay_every epochs
	lr_tmp = lr * (0.5 ** (epoch // learning_rate_decay_every))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_tmp
	return lr_tmp

def check_restart_conditions(params):

	# Check for the status file corresponding to the model
	status_file = os.path.join(params['checkpoint_path'], 'status.json')
	if os.path.exists(status_file):
		with open(status_file, 'r') as f:
			status = json.load(f)
		params['resume_from_epoch'] = status['epoch']
	else:
		params['resume_from_epoch'] = 0
	return params

def write_status(params, epoch):

	status_file = os.path.join(params['checkpoint_path'], 'status.json')
	status = {
		'epoch': epoch,
	}

	with open(status_file, 'w') as f:
		json.dump(status, f, indent=4)

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

def train(models, train_dataset, val_dataset, params, extra_params):
	
	params = check_restart_conditions(params)
	
	# Construct Data loader
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=params['batch_size'], num_workers=1)

	if params['use_gpu'] and torch.cuda.is_available():
		print('Initialized Cuda Models')
		
		for mname in models:
			models[mname] = models[mname].cuda()

	if params['resume_from_epoch'] >= 1:
		
		print('Loading Old model')
		
		load_model_dir = os.path.join(params['checkpoint_path'])
		
		print('Loading model files from folder: %s' % load_model_dir)
		
		models = load_models(models, params, load_model_dir)

	# Loss and optimizers
	criterion = nn.CrossEntropyLoss()

	optimizer_parameter_group = [ { 'params': models[m].parameters()} for m in models.keys() ]

	if params['optim'] == 'sgd':
		optimizer = torch.optim.SGD(optimizer_parameter_group,
									lr=params['learning_rate'],
									momentum=params['momentum'])
	elif params['optim'] == 'rmsprop':
		optimizer = torch.optim.RMSprop(optimizer_parameter_group,
										lr=params['learning_rate'],
										alpha=params['optim_alpha'],
										eps=params['optim_epsilon'],
										momentum=params['momentum'])
	elif params['optim'] == 'adam':
		optimizer = torch.optim.Adam(optimizer_parameter_group,
					 eps=params['optim_epsilon'],
					 lr=params['learning_rate'])
	
	elif params['optim'] == 'rprop':
		optimizer = torch.optim.Rprop(optimizer_parameter_group,
									 lr=params['learning_rate'])
	else:
		raise('Unsupported optimizer: \'%s\'' % (params['optim']))
		return None

	# Start training
	all_loss_store = []
	loss_store = []
	val_acc_store = []
	val_loss_store = []

	lr_cur = params['learning_rate']

	if params['use_roi'] or params['load_roi']:
		roi_save_file = h5py.File(params['roi_save_file'])

	# Train loop
	for epoch in range(params['resume_from_epoch'], params['epochs']+1):

		# Call train() on all models for training
		for m in models.keys():
			models[m].train()
		
		if epoch > params['learning_rate_decay_start']:
			lr_cur = adjust_learning_rate(optimizer, epoch - 1 - params['learning_rate_decay_start'] + params['learning_rate_decay_every'],
										  params['learning_rate'], params['learning_rate_decay_every'])
		print('Epoch: %d | lr: %f' % (epoch, lr_cur))

		running_loss = 0.0
		
		#if epoch > 0:
		#	params['load_roi'] = True
		#print(train_loader.params['load_roi'])
		for i, batch in enumerate(train_loader):
			
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
					
			optimizer.zero_grad()
			
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

			loss = criterion(output, answers)
			
			all_loss_store += [[epoch, loss.item()]]
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if not (i+1) % params['losses_log_every']:
				print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
					epoch, params['epochs'], i+1,
					train_dataset.__len__()//params['batch_size'], loss.item()
					))

		accuracies = []
		# Call train() on all models for training
		for m in models:
			models[m].eval()
			
		if params['use_gpu'] and torch.cuda.is_available():
			pred = lambda x: np.argmax(x.cpu().detach().numpy(), axis=1)
		else:    
			pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

		running_val_loss = 0.0
		for i, batch in enumerate(val_loader):

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

			optimizer.zero_grad()
			
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

		val_loss_store += [[epoch, running_val_loss]]
		val_acc_store += [[epoch, np.mean(accuracies)]]

		print('Epoch [%d/%d], Val Accurracy: %.4f' % (epoch, params['epochs'], np.mean(accuracies)))        

		print("Saving models")
		model_dir = os.path.join(params['checkpoint_path'])
		
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		
		torch.save(models['ques_model'].state_dict(), os.path.join(model_dir, 'question_model.pkl'))
		torch.save(models['att_model'].state_dict(), os.path.join(model_dir, 'attention_model.pkl'))

		if not params['load_roi']:
			torch.save(models['img_model'].state_dict(), os.path.join(model_dir, 'image_model.pkl'))
			if params['use_roi']:
				torch.save(models['roi_model'].state_dict(), os.path.join(model_dir, 'roi_model.pkl'))
		
		if params['use_text']:
			torch.save(models['text_model'].state_dict(), os.path.join(model_dir, 'text_model.pkl'))
		
		write_status(params, epoch)
		loss_store += [[epoch, running_loss]]
		print('Epoch %d | Loss: %.4f | lr: %f'%(epoch, running_loss, lr_cur))

		# torch.save(question_model.state_dict(), 'question_model'+str(epoch)+'.pkl')
		print("Saving all losses to file")
		np.savetxt(os.path.join(params['checkpoint_path'], 'all_loss_store.txt'), np.array(all_loss_store), fmt='%f')
		np.savetxt(os.path.join(params['checkpoint_path'], 'loss_store.txt'), np.array(loss_store), fmt='%f')
		np.savetxt(os.path.join(params['checkpoint_path'], 'val_loss_store.txt'), np.array(val_loss_store), fmt='%f')
		np.savetxt(os.path.join(params['checkpoint_path'], 'val_acc_store.txt'), np.array(val_acc_store), fmt='%f')
		
		print(loss_store)

