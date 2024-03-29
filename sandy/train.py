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

def main(args):
    
    params = vars(args)

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

def fetch_args(parser):

    parser.add_argument('--data-dir', type=str, required=True, help="The Path of the dataset containing the images, QA and metadata")
    parser.add_argument('--no-debug', dest='debug', default=True, action='store_false', help="Turn off printing off logs all the modules")
    parser.add_argument('--split', type=str, default='train', help="The dataset split we want to work with for training the model")
    parser.add_argument('--san', dest='use_dyn_dict', default=True, action="store_false", help="To train original SAN model, turn off the use of dynamic dictionary")
    parser.add_argument('--idx_dir', default='gen/')
    parser.add_argument('--resume_from_epoch', type=int, dest='resume_from_epoch', default=0, help='Resume from which epoch')
    parser.add_argument('--small_train', dest='small_train', default=False, action='store_true', help='For training on a small training set')
    parser.add_argument('--pct', default=1, type=int, help="Percentage of data to be used")

    # TODO: To support resuming from previous checkpoint
    # parser.add_argument('--warm-restart', )

    # Options
    parser.add_argument('--feature_type', default='Resnet152', help='VGG16 or Resnet152')
    parser.add_argument('--emb_size', default=300, type=int, help='the size after embedding from onehot')
    parser.add_argument('--hidden_size', default=1024, type=int, help='the hidden layer size of the model')
    parser.add_argument('--rnn_size', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--att_size', default=512, type=int, help='size of attention vector which refer to k in paper')
    parser.add_argument('--batch_size', default=16, type=int, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--output_size', default=1000, type=int, help='number of output answers')
    parser.add_argument('--rnn_layers', default=2, type=int, help='number of the rnn layer')
    parser.add_argument('--img_seq_size', default=196, type=int, help='number of feature regions in image')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio in network')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs to run')

    # Optimization
    parser.add_argument('--optim', default='adam', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--learning_rate_decay_start', default=10, type=int, help='at what epoch to start decaying learning rate?')
    parser.add_argument('--learning_rate_decay_every', default=10, type=int, help='every how many epoch thereafter to drop LR by 0.1?')
    parser.add_argument('--optim_alpha', default=0.99, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--optim_beta', default=0.995, type=float, help='beta used for adam')
    parser.add_argument('--optim_epsilon', default=1e-8, type=float, help='epsilon that goes into denominator in rmsprop')
    parser.add_argument('--max_iters', default=-1, type=int, help='max number of iterations to run for (-1 = run forever)')
    parser.add_argument('--iterPerEpoch', default=1250, type=int, help=' no. of iterations per epoch')

    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_every', default=500, type=int, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', default='train_model/', help='folder to save checkpoints into (empty = this folder)')

    # Visualization
    parser.add_argument('--losses_log_every', default=10, type=int, help='How often do we save losses, for inclusion in the progress dump? (0 = disable)')

    # misc
    parser.add_argument('--use_gpu', default=1, type=int, help='to use gpu or not to use, that is the question')
    parser.add_argument('--id', default='1', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--gpuid', default=-1, type=int, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', default=1234, type=int, help='random number generator seed to use')
    parser.add_argument('--print_params', default=1, type=int, help='pass 0 to turn off printing input parameters')

    args = parser.parse_args()
    params = vars(args)                     # convert to ordinary dict
    
    if params['print_params']:
        print('parsed input parameters:')
        print (json.dumps(params, indent = 2))

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = fetch_args(parser)
    main(args)
