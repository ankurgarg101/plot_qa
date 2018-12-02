"""
Main module that performs the training and evaluation process
"""
import argparse
from utils.dataset import PlotDataset
from models import build_models
import json
from trainer import train
import os
from evaluator import eval_model
from visualize import visualize_model

def fetch_args(parser):

	parser.add_argument('--data-dir', type=str, required=True, help="The Path of the dataset containing the images, QA and metadata")
	parser.add_argument('--no-debug', dest='debug', default=True, action='store_false', help="Turn off printing off logs all the modules")
	parser.add_argument('--split', type=str, default='train', help="The dataset split we want to work with for training the model")
	parser.add_argument('--san', dest='use_dyn_dict', default=True, action="store_false", help="To train original SAN model, turn off the use of dynamic dictionary")
	parser.add_argument('--idx_dir', default='gen/')
	parser.add_argument('--resume_from_epoch', type=int, dest='resume_from_epoch', default=0, help='Resume from which epoch')
	parser.add_argument('--small_train', dest='small_train', default=False, action='store_true', help='For training on a small training set')

	parser.add_argument('--pct', default=1, type=int, help="Percentage of data to be used")
	parser.add_argument('--random', default=False, action="store_true", help="Randomly select the data while selecting a subset")

	# TODO: To support resuming from previous checkpoint
	# parser.add_argument('--warm-restart', )

	# Options
	parser.add_argument('--feature_type', default='VGG16', help='VGG16 or Resnet152')
	parser.add_argument('--emb_size', default=100, type=int, help='the size after embedding from onehot')
	parser.add_argument('--hidden_size', default=256, type=int, help='the hidden layer size of the question embedding model')
	parser.add_argument('--rnn_size', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
	parser.add_argument('--att_size', default=512, type=int, help='size of attention vector which refer to k in paper')
	parser.add_argument('--batch_size', default=4, type=int, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
	parser.add_argument('--output_size', default=1000, type=int, help='number of output answers')
	parser.add_argument('--rnn_layers', default=2, type=int, help='number of the rnn layer')
	parser.add_argument('--img_seq_size', default=196, type=int, help='number of feature regions in image')
	parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio in network')
	parser.add_argument('--epochs', default=2, type=int, help='Number of epochs to run')
	parser.add_argument('--roi_crop', default=7, type=int, help='Crop Dim Size')

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

	parser.add_argument('--roi_save_file', default='roi_dump.h5', help='roi dump')

	#ablation
	parser.add_argument('--use_text', dest = 'use_text', default=False, action = 'store_true', help = 'Use Text supervision in the images')
	parser.add_argument('--use_roi', dest = 'use_roi', default=False, action = 'store_true', help = 'Use ROI features for the bbox regions')
	parser.add_argument('--use_pos', dest = 'use_pos', default=False, action = 'store_true', help = 'Use positional features from bboxes for text and bars')
	parser.add_argument('--load_roi', dest = 'load_roi', default=False, action = 'store_true', help = 'Load and use precomputed positional features from bboxes for text and bars')
	parser.add_argument('--use_global_img', default=False, action='store_true', help="Use global Image features as well")
	parser.add_argument('--n_heads', default=1, type=int, help='Number of attention heads')
	parser.add_argument('--dot_product_att', default=False, action="store_true", help="Dot product or bahnadau")

	# visualization 
	parser.add_argument('--vis', dest = 'vis', default=False, action = 'store_true', help = 'Visualize attention weights')  
	parser.add_argument('--vis_idx', dest = 'vis_idx', default=0, type=int, help = 'Index of question to visualize')    
	parser.add_argument('--vis_dir', dest = 'vis_dir', default='vis/', help = 'Directory to place visualizations')  
	 # N Heads

	args = parser.parse_args()
	params = vars(args)                     # convert to ordinary dict
	
	if params['print_params']:
		print('parsed input parameters:')
		print (json.dumps(params, indent = 2))

	return params

def get_extra_params(train_dataset):

	extra_params = {}

	extra_params['max_num_bars'] = train_dataset.max_num_bars
	extra_params['max_num_text'] = train_dataset.max_num_text
	extra_params['ques_vocab_size'] = train_dataset.ques_vocab_size
	extra_params['ans_vocab_size'] = train_dataset.ans_vocab_size
	extra_params['text_vocab_size'] = train_dataset.ques_vocab_size
	extra_params['max_ques_seq_len'] = train_dataset.max_ques_len

	if params['feature_type'] == 'VGG16':
		extra_params['sfeat_img'] = 512
		# For using global image features as well in the model
		extra_params['img_feat_size'] = 512
	else:
		extra_params['sfeat_img'] = 2048
		# For using global image features as well in the model
		extra_params['img_feat_size'] = 2048

	if params['use_pos']:
		extra_params['sfeat_text'] = params['emb_size'] + train_dataset.n_text_types + 4
		
		if params['use_roi']:
			extra_params['sfeat_img'] += 4
	else:
		extra_params['sfeat_text'] = params['emb_size'] + train_dataset.n_text_types
	extra_params['sfeat_ques'] = params['hidden_size']

	if not params['use_global_img']:
		extra_params['img_feat_size'] = None

	return extra_params

def save_json(data, ddir, fname):

	if not os.path.exists(ddir):
		os.makedirs(ddir)

	with open(os.path.join(ddir, fname), 'w') as f:
		json.dump(data, f, indent=4)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	params = fetch_args(parser)

	if params['vis']:	
		params['split'] = 'val_easy'	
		dataset = PlotDataset(params, 'val_easy')	
			
		extra_params = get_extra_params(dataset)	
		models = build_models(params, extra_params)	
		visualize_model(models, dataset, params, extra_params, int(params['vis_idx']))	
		exit() # Abrupt exit
		 
	if params['split'] == 'train':
		
		train_dataset = PlotDataset(params, 'train')
		val_dataset = PlotDataset(params, 'val_easy')   

		extra_params = get_extra_params(train_dataset)

		save_json(params, params['checkpoint_path'], 'params.json')
		save_json(extra_params, params['checkpoint_path'], 'extra_params.json')

		models = build_models(params, extra_params)
	
		train(models, train_dataset, val_dataset, params, extra_params)

	else:

		print(params['split'])
		val_easy_dataset = PlotDataset(params, 'val')

		extra_params = get_extra_params(val_easy_dataset)
		
		models = build_models(params, extra_params)
		eval_model(models, val_easy_dataset, params, extra_params)

