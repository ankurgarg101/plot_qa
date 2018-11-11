"""
Module that implements uses the PyTorch dataset interface to load the dataset and process the data samples needed for training or evaluation.
"""

from os import path
from glob import glob
import numpy as np
from skimage import io
import json

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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

		with open(path.join(self.meta_data_dir, '{}_metadata.json'.format(self.split)), 'r') as metadata_file:
			self.metadata = json.load(metadata_file)

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

		# Process the Questions and Answers to Construct the dictionaries

		self._process_answers()

	def _process_answers(self):

		# Check for split to decide whether to label words as unknown or not

	def __len__(self):
		return len(self.idx2qid)

	def __getitem__(self, idx):

		"""
		Return the item according to the index
		"""
		pass