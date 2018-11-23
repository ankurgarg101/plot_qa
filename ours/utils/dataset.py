"""
Module that implements uses the PyTorch dataset interface to load the dataset and process the data samples needed for training or evaluation.
"""

from os import path
from glob import glob
import numpy as np
import skimage
from skimage import io
import json
import torch
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

	def __init__(self, params, split):

		super(PlotDataset, self).__init__()

		self.data_dir = params['data_dir']
		self.img_dir = path.join(self.data_dir, 'images')
		self.meta_data_dir = path.join(self.data_dir, 'metadata')
		self.qa_dir = path.join(self.data_dir, 'qa')

		self.split = split
		self.use_dyn_dict = params['use_dyn_dict']
		self.params = params

		assert path.exists(self.img_dir)
		assert path.exists(self.meta_data_dir)
		assert path.exists(self.qa_dir)

		if params['debug']:
			print('Reading data from Location {}'.format(self.data_dir))

		# Verify the split provided		
		if self.split not in ['train', 'val_easy', 'val_hard']:
			raise( "Invalid Dataset Split mentioned. Please enter one of \{ train, val_easy, val_hard \}")

		# Read the data corresponding to self.split
		#self.img_filenames = glob(path.join(self.img_dir, 'bar_{}_*-img.png'.format(self.split)))

		with open(path.join(self.qa_dir, '{}_qa_tok.json'.format(self.split)), 'r') as qa_file:
			qa_data = json.load(qa_file)

		with open(path.join(self.meta_data_dir, '{}_metadata_lbl.json'.format(self.split)), 'r') as metadata_file:
			metadata = json.load(metadata_file)

		# divide the data according to the percentage specified in params
		num_ex = int(params['pct']*len(qa_data)/100)
		print ('Using only %d questions'%num_ex)
		qa_data = qa_data[:num_ex]

		if params['small_train']:
			num_ex = 100
			print ('Using only %d questions'%num_ex)
			qa_data = qa_data[: num_ex]

		# Create a map of Metadata indexed by image instead of a list
		self.metadata_dict = {}
		
		# Create a Map from Question Id to Idx for __getitem__ to work correctly
		self.qid2idx = {}
		self.idx2qid = {}
		self.qa_dict = {}

		idx = 0
		for qas in qa_data:
			self.idx2qid[idx] = qas['question_id']
			self.qid2idx[qas['question_id']] = idx
			self.qa_dict[qas['question_id']] = qas
			self.metadata_dict[qas['image']] = {}
			idx += 1

		for mt in metadata:
			if mt['image'] in self.metadata_dict:
				self.metadata_dict[mt['image']] = mt 
		
		# Index the Questions and Answers
		self.index_answers()
		self.index_questions()

		self.ans_vocab_size = len(self.ans_indexer)
		self.ques_vocab_size = len(self.ques_indexer)

		self.max_num_text = 30
		self.max_num_bars = 30
		self.text_types = ['legend_heading', 'legend', 'value_ticks', 'title', 'value_heading', 'label_ticks']
		self.n_text_types = len(self.text_types)
		self.org_h = 448
		self.org_w = 448
		self.scale_ratio = 448/14

		# Added a normalization transform according to torchvision documentation
		self.img_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		if params['debug']:
			print('Read {} Question-Answer Pairs'.format(len(self.idx2qid)))
			print('Max Ques Len: {}'.format(self.max_ques_len))
		
	def index_questions(self):

		# Creates the Index for Questions

		self.max_ques_len = 0

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
					self.ques_indexer.get_index('bbox_%02d'%bidx)

			for qid in self.qa_dict:

				qa = self.qa_dict[qid]
				question_wrds = qa['question_tok']

				self.max_ques_len = max(self.max_ques_len, len(question_wrds))

				if self.use_dyn_dict:
					
					# Check if the word exists in any bounding box

					bbox_txt_metadata = self.metadata_dict[qa['image']]['texts']

					for wrd in question_wrds:

						isPresentInBbox = False
						
						for bbox_txt in bbox_txt_metadata:

							if bbox_txt['text'] == wrd:

								isPresentInBbox = True
								break

						if not isPresentInBbox:
							self.ques_indexer.get_index(wrd) 

				else:
					
					# If not using dynamic dictionary, add all the words in the question in the dictionary

					for	wrd in question_wrds:
						self.ques_indexer.get_index(wrd)
					

			# Save the Indexer to be used for 
			self.ques_indexer.dump(path.join(self.params['idx_dir'], 'question_indexer_{}.json'.format(suffix)))
		else:

			assert path.exists(path.join(self.params['idx_dir'], 'question_indexer_{}.json'.format(suffix)))

			# Just Load the Indexer from the given file path.
			self.ques_indexer = Indexer(path.join(self.params['idx_dir'], 'question_indexer_{}.json'.format(suffix)))

			# Compute the self.max_ques_len

			self.max_ques_len = max( [ len(qas['question_tok']) for qas in self.qa_dict.values() ] )


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
			self.ans_indexer.dump(path.join(self.params['idx_dir'], 'answer_indexer_{}.json'.format(suffix)))

		else:

			assert path.exists(path.join(self.params['idx_dir'], 'answer_indexer_{}.json'.format(suffix)))

			# Just Load the Indexer from the given file path.
			self.ans_indexer = Indexer(path.join(self.params['idx_dir'], 'answer_indexer_{}.json'.format(suffix)))

	def tokenize(self, qas):

		# Tokenize the answer first

		# Prevent adding the words to indexer
		addToIndexer = False

		question_tok = np.ones((self.max_ques_len), dtype=np.int)*self.ques_indexer.get_index(pad_token, addToIndexer)
		
		answer = qas['answer']
		question_wrds = qas['question_tok']
		question_len = len(question_wrds)
		answer_tok = None

		if self.use_dyn_dict:

			image_metadata = self.metadata_dict[qas['image']]

			if len(qas['answer_bbox']) > 0:
				# find the bbox in the metadata
				for mt in image_metadata['texts']:
					if qas['answer_bbox'] == mt['bbox']:
						bidx = mt['idx']
						answer_idx = self.ans_indexer.get_index('bbox_%02d'%bidx, addToIndexer)
						assert answer_idx != -1

						answer_tok = np.array([answer_idx], dtype=np.int)
						break
			else:

				answer_idx = self.ans_indexer.get_index(answer, addToIndexer)

				if answer_idx == -1:
					answer_tok = np.array([self.ans_indexer.get_index(unk_token, addToIndexer)], dtype=np.int)
				else:
					answer_tok = np.array([answer_idx], dtype=np.int)


			# Check if the word exists in any bounding box
			bbox_txt_metadata = self.metadata_dict[qas['image']]['texts']

			for i, wrd in enumerate(question_wrds):

				isPresentInBbox = False
						
				for bbox_txt in bbox_txt_metadata:

					if bbox_txt['text'] == wrd:
						isPresentInBbox = True
						bidx = bbox_txt['idx']
						qidx = self.ques_indexer.get_index('bbox_%02d'%bidx, addToIndexer)
						assert qidx != -1

						question_tok[i] = qidx
						break

				if not isPresentInBbox:
					qidx = self.ques_indexer.get_index(wrd, addToIndexer)

					if qidx == -1:
						question_tok[i] = self.ques_indexer.get_index(unk_token, addToIndexer)
					else:
						question_tok[i] = qidx
		else:

			# Tokenize the anwer
			answer_idx = self.ans_indexer.get_index(answer, addToIndexer)

			if answer_idx == -1:
				answer_tok = np.array([self.ans_indexer.get_index(unk_token, addToIndexer)], dtype=np.int)
			else:
				answer_tok = np.array([answer_idx], dtype=np.int)

			# Tokenize the question

			for i, wrd in enumerate(question_wrds):

				wrd_idx = self.ques_indexer.get_index(wrd, addToIndexer)

				if wrd_idx == -1:
					question_tok[i] = self.ques_indexer.get_index(unk_token, addToIndexer)
				else:
					question_tok[i] = wrd_idx


		return question_tok, question_len, answer_tok

	
	def get_bbox_data(self, metadata):

		addToIndexer = False
		bar_bboxes = np.zeros((self.max_num_bars, 4), dtype=np.float32)
		text_bboxes = np.zeros((self.max_num_text, 4), dtype=np.float32)
		text_vals = np.zeros((self.max_num_text), dtype=np.int)
		text_bbox_type = np.zeros((self.max_num_text, self.n_text_types))

		bidx = 0
		# First fill the bar_bboxes
		for br in metadata['bars']['bboxes']:
			for box in br:
				center_x, center_y, w, h = box[0], box[1], box[2], box[3]
				x1 = (center_x - w/2) / (self.org_w - 1)
				x2 = (center_x + w/2) / (self.org_w - 1)
				y1 = (center_y - h/2) / (self.org_h - 1)
				y2 = (center_y + h/2) / (self.org_h - 1)

				bar_bboxes[bidx][0] = x1 / self.scale_ratio
				bar_bboxes[bidx][1] = y1 / self.scale_ratio
				bar_bboxes[bidx][2] = x2 / self.scale_ratio
				bar_bboxes[bidx][3] = y2 / self.scale_ratio

				bidx += 1

		txt_len = 0
		for txt in metadata['texts']:
			
			tidx = txt['idx']
			text_bbox_type[tidx][self.text_types.index(txt['text_function'])] = 1

			text_vals[tidx] = self.ques_indexer.get_index('bbox_%02d'%tidx, addToIndexer)

			box = txt['bbox']

			center_x, center_y, w, h = box[0], box[1], box[2], box[3]
			x1 = (center_x - w/2) / (self.org_w - 1)
			x2 = (center_x + w/2) / (self.org_w - 1)
			y1 = (center_y - h/2) / (self.org_h - 1)
			y2 = (center_y + h/2) / (self.org_h - 1)

			text_bboxes[tidx][0] = x1 / self.scale_ratio
			text_bboxes[tidx][1] = y1 / self.scale_ratio
			text_bboxes[tidx][2] = x2 / self.scale_ratio
			text_bboxes[tidx][3] = y2 / self.scale_ratio

			txt_len = max(txt_len, tidx)

		bar_len = bidx
		txt_len = txt_len + 1

		return bar_bboxes, text_bboxes, text_vals, text_bbox_type, bar_len, txt_len

	def __len__(self):
		return len(self.idx2qid)

	def __getitem__(self, idx):

		"""
		Return the item according to the index
		"""
		
		# Get the question Id to process
		question_id = self.idx2qid[idx]
		
		# First, read the image
		image_name = self.qa_dict[question_id]['image']
		image_path = path.join(self.img_dir, image_name)
		rgba_image = io.imread(image_path)
		rgb_image = skimage.color.rgba2rgb(rgba_image)
		img = self.img_transform(rgb_image)

		question_tok, question_len, answer_tok = self.tokenize(self.qa_dict[question_id])

		bar_bboxes, text_bboxes, text_vals, text_types, bar_len, txt_len = self.get_bbox_data(self.metadata_dict[image_name])

		return {
			'image': img,
			'ques': torch.as_tensor(question_tok, dtype=torch.long),
			'ques_len': question_len,
			'ans': torch.as_tensor(answer_tok, dtype=torch.long),
			'bar_len': bar_len,
			'txt_len': txt_len,
			'bar_bboxes': torch.as_tensor(bar_bboxes, dtype=torch.float),
			'text_bboxes': torch.as_tensor(text_bboxes, dtype=torch.float),
			'text_vals': torch.as_tensor(text_vals, dtype=torch.long),
			'text_types': torch.as_tensor(text_types, dtype=torch.long)
		}
