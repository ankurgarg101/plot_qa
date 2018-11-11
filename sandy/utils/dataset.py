"""
Module that implements uses the PyTorch dataset interface to load the dataset and process the data samples needed for training or evaluation.
"""

from os import path
from glob import glob
import numpy as np
from skimage import io
import json
import nltk

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .indexer import Indexer

pad_token = '<pad>'
unk_token = '<unk>'
max_bboxes = 30

class PlotDataset(Dataset):

	"""
	Dataset class that loads the DVQA dataset
	"""

	def __init__(self, args):

		super(PlotDataset, self).__init__()

		self.data_dir = args.data_dir
		self.img_dir = path.join(self.data_dir, 'images')
		self.meta_data_dir = path.join(self.data_dir, 'metadata')
		self.qa_dir = path.join(self.data_dir, 'qa')

		self.split = args.split
		self.use_dyn_dict = args.use_dyn_dict
		self.args = args

		assert path.exists(self.img_dir)
		assert path.exists(self.meta_data_dir)
		assert path.exists(self.qa_dir)

		if args.debug:
			print('Reading data from Location {}'.format(self.data_dir))

		# Verify the split provided		
		if self.split not in ['train', 'val_easy', 'val_hard']:
			raise( "Invalid Dataset Split mentioned. Please enter one of \{ train, val_easy, val_hard \}")

		# Read the data corresponding to self.split
		self.img_filenames = glob(path.join(self.img_dir, 'bar_{}_*-img.png'.format(self.split)))

		with open(path.join(self.qa_dir, '{}_qa.json'.format(self.split)), 'r') as qa_file:
			qa_data = json.load(qa_file)

		with open(path.join(self.meta_data_dir, '{}_metadata_lbl.json'.format(self.split)), 'r') as metadata_file:
			metadata = json.load(metadata_file)

		# Create a map of Metadata indexed by image instead of a list
		self.metadata_dict = {}
		for mt in metadata:
			self.metadata_dict['image'] = mt 

		# Create a Map from Question Id to Idx for __getitem__ to work correctly
		self.qid2idx = {}
		self.idx2qid = {}
		self.qa_dict = {}

		idx = 0
		for qas in qa_data:
			self.idx2qid[idx] = qas['question_id']
			self.qid2idx[qas['question_id']] = idx
			self.qa_dict[qas['question_id']] = qas
			idx += 1

		if args.debug:
			print('Read {} Question-Answer Pairs'.format(len(self.idx2qid)))

		# Index the Questions and Answers
		self.index_answers()
		self.index_questions()

	def index_questions(self):

		# Creates the Index for Questions
		suffix = 'san'
		if self.use_dyn_dict:
			suffix += 'dy'

		if self.split == 'train':
			self.ques_indexer = Indexer()
			
			# Add the Pad and UNK tokens at the start
			self.ques_indexer.get_index(pad_token)
			self.ques_indexer.get_index(unk_token)

			# If using the dynamic dictionary, add the bounding box token ids at the start of the dictionary.

			if self.use_dyn_dict:
				for bidx in range(max_bboxes):
					self.ans_indexer.get_index('bbox_%02d'%bidx)

			for qid in self.qa_dict:

				qa = self.qa_dict[qid]
				question_wrds = nltk.word_tokenize( qa['question'] )

				if self.use_dyn_dict:
					
					# Check if the word exists in any bounding box

					bbox_txt_metadata = self.metadata_dict[qa['image']]['texts']

					for wrd in question_wrds:

						isPresentInBbox = False
						
						for bbox_txt in bbox_txt_metadata:

							if bbox_txt['text'] == wrd:

								isPresentInBbox = True

						if not isPresentInBbox:
							self.ques_indexer.get_index(wrd) 

				else:
					# If not using dynamic dictionary, add all the words in the question in the dictionary
					
					for	wrd in question_wrds:
						self.ques_indexer.get_index(wrd)
					

			# Save the Indexer to be used for 
			self.ques_indexer.dump(path.join(self.args.idx_dir, 'question_indexer_{}.json'.format(suffix)))
		else:

			assert path.exists(path.join(self.args.idx_dir, 'question_indexer_{}.json'.format(suffix)))

			# Just Load the Indexer from the given file path.
			self.ques_indexer = Indexer(path.join(self.args.idx_dir, 'question_indexer_{}.json'.format(suffix)))

	def index_answers(self):

		# Creates the indexer for answers

		suffix = 'san'
		if self.use_dyn_dict:
			suffix += 'dy'

		if self.split == 'train':

			self.ans_indexer = Indexer()

			# Add the Pad and UNK tokens at the start
			self.ans_indexer.get_index(pad_token)
			self.ans_indexer.get_index(unk_token)

			# If using the dynamic dictionary, add the bounding box token ids at the start of the dictionary.

			if self.use_dyn_dict:
				for bidx in range(max_bboxes):
					self.ans_indexer.get_index('bbox_%02d'%bidx)

			for qid in self.qa_dict:

				qa = self.qa_dict[qid]

				if self.use_dyn_dict:
					# Check if answer belongs to a bounding box or not
					if len(qa['answer_bbox']) == 0:
						self.ans_indexer.get_index(qa['answer'])
				else:
					
					# If not using dynamic dictionary, add all answers in the dictionary
					
					self.ans_indexer.get_index(qa['answer'])

			# Save the Indexer to be used for 
			self.ans_indexer.dump(path.join(self.args.idx_dir, 'answer_indexer_{}.json'.format(suffix)))

		else:

			assert path.exists(path.join(self.args.idx_dir, 'answer_indexer_{}.json'.format(suffix)))

			# Just Load the Indexer from the given file path.
			self.ans_indexer = Indexer(path.join(self.args.idx_dir, 'answer_indexer_{}.json'.format(suffix)))


	def __len__(self):
		return len(self.idx2qid)

	def __getitem__(self, idx):

		"""
		Return the item according to the index
		"""
		pass