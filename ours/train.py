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
from utils.dataset import PlotDataset
from torch.autograd import Variable

from models.img_emb import ImageEmbedding
from models.ques_emb import QuestionEmbedding
from models.san import SAN


def adjust_learning_rate(optimizer, epoch, lr, learning_rate_decay_every):
    
    # Sets the learning rate to the initial LR decayed by 10 every learning_rate_decay_every epochs
    lr_tmp = lr * (0.5 ** (epoch // learning_rate_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp
    return lr_tmp

def cycle(seq):
    while True:
        for elem in seq:
            yield elem

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

def train(params):
    
    params = check_restart_conditions(params)
    # Construct Data loader
    
    train_dataset = PlotDataset(args, 'train')
    val_dataset = PlotDataset(args, 'val_easy')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=params['batch_size'], num_workers=8)

    # Construct NN models
    vocab_size = train_dataset.ques_vocab_size
    output_size = train_dataset.ans_vocab_size

    question_model = QuestionEmbedding(vocab_size, params['emb_size'],
                                       params['hidden_size'], params['rnn_size'],
                                       params['rnn_layers'], params['dropout'],
                                       train_dataset.max_ques_len, params['use_gpu'])

    image_model = ImageEmbedding(params['hidden_size'], params['feature_type'])

    attention_model = SAN(params['hidden_size'], params['att_size'],
                                params['img_seq_size'], output_size)

    if params['use_gpu'] and torch.cuda.is_available():
        print('Initialized Cuda Models')
        question_model.cuda()
        image_model.cuda()
        attention_model.cuda()

    if params['resume_from_epoch'] >= 1:
        print('Loading Old model')
        #load_model_dir = os.path.join(params['checkpoint_path'], str(params['resume_from_epoch']-1))
        load_model_dir = os.path.join(params['checkpoint_path'])
        print('Loading model files from folder: %s' % load_model_dir)
        question_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'question_model.pkl')))
        image_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'image_model.pkl')))
        attention_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'attention_model.pkl')))

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()

    optimizer_parameter_group = [
            {'params': question_model.parameters()},
            {'params': image_model.parameters()},
            {'params': attention_model.parameters()}
            ]
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

    # Call train() on all models for training
    question_model.train()
    image_model.train()
    attention_model.train()

    # Train loop
    for epoch in range(params['resume_from_epoch'], params['epochs']+1):

        if epoch > params['learning_rate_decay_start']:
            lr_cur = adjust_learning_rate(optimizer, epoch - 1 - params['learning_rate_decay_start'] + params['learning_rate_decay_every'],
                                          params['learning_rate'], params['learning_rate_decay_every'])
        print('Epoch: %d | lr: %f' % (epoch, lr_cur))

        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch['image']
            questions = batch['ques']
            ques_lens = batch['ques_len']
            answers = batch['ans']

            # Sort the examples in reverse order of sentence length
            _, sort_idxes = torch.sort(ques_lens, descending=True)
            images = images[sort_idxes, :, :, :]
            questions = questions[sort_idxes, :]
            ques_lens = ques_lens[sort_idxes]
            answers = answers[sort_idxes, :]
            answers = answers.squeeze(1)
            
            # print (images)
            # print (questions)
            # print (ques_lens)
            # print (answers)

            if (params['use_gpu'] and torch.cuda.is_available()):
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()

            optimizer.zero_grad()
            img_emb = image_model.forward(images)
            ques_emb = question_model.forward(questions, ques_lens)
            output = attention_model.forward(ques_emb, img_emb)

            loss = criterion(output, answers)
            
            all_loss_store += [loss.item()]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if not (i+1) % params['losses_log_every']:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                    epoch, params['epochs'], i+1,
                    train_dataset.__len__()//params['batch_size'], loss.item()
                    ))

        accuracies = []
        
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

            # Sort the examples in reverse order of sentence length
            _, sort_idxes = torch.sort(ques_lens, descending=True)
            images = images[sort_idxes, :, :, :]
            questions = questions[sort_idxes, :]
            ques_lens = ques_lens[sort_idxes]
            answers = answers[sort_idxes, :]
            answers = answers.squeeze(1)
            
            # print (images)
            # print (questions)
            # print (ques_lens)
            # print (answers)

            if (params['use_gpu'] and torch.cuda.is_available()):
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()

            img_emb = image_model.forward(images)
            ques_emb = question_model.forward(questions, ques_lens)
            output = attention_model.forward(ques_emb, img_emb)
            val_loss = criterion(output, answers)
            running_val_loss += val_loss.item()

            if (params['use_gpu'] and torch.cuda.is_available()):
                answers = answers.cpu()

            output_preds = pred(output)
            accuracies.extend( output_preds ==  answers.detach().numpy())

        val_loss_store += [running_val_loss]
        val_acc_store += [np.mean(accuracies)]

        print('Epoch [%d/%d], Val Accurracy: %.4f' % (epoch, params['epochs'], np.mean(accuracies)))        

        print("Saving models")
        model_dir = os.path.join(params['checkpoint_path'])
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save(question_model.state_dict(), os.path.join(model_dir, 'question_model.pkl'))
        torch.save(image_model.state_dict(), os.path.join(model_dir, 'image_model.pkl'))
        torch.save(attention_model.state_dict(), os.path.join(model_dir, 'attention_model.pkl'))
        write_status(params, epoch)
        loss_store += [running_loss]
        print('Epoch %d | Loss: %.4f | lr: %f'%(epoch, running_loss, lr_cur))

        # torch.save(question_model.state_dict(), 'question_model'+str(epoch)+'.pkl')
        print("Saving all losses to file")
        np.savetxt(os.path.join(params['checkpoint_path'], 'all_loss_store.txt'), np.array(all_loss_store), fmt='%f')
        np.savetxt(os.path.join(params['checkpoint_path'], 'loss_store.txt'), np.array(loss_store), fmt='%f')
        np.savetxt(os.path.join(params['checkpoint_path'], 'val_loss_store.txt'), np.array(val_loss_store), fmt='%f')
        np.savetxt(os.path.join(params['checkpoint_path'], 'val_acc_store.txt'), np.array(val_acc_store), fmt='%f')
        print(loss_store)

