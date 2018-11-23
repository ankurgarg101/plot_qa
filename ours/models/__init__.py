"""
Functions to build the correct models based on the specified parameters.
"""

from .img_emb import ImageEmbedding
from .ques_emb import QuestionEmbedding
from .san import SAN
from .roi import RoIFeats
from .text_emb import TextEmbedding

def build_models(params, extra_params):

	# Models that will be added irrespective of the other flags

	models = []

	question_model = QuestionEmbedding(extra_params['ques_vocab_size'], params['emb_size'], params['hidden_size'], params['rnn_size'], params['rnn_layers'], params['dropout'], extra_params['max_ques_seq_len'], params['use_gpu'])

	attention_model = SAN(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'])
	
	if params['use_text']:
		attention_model = SAN(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'], extra_params['sfeat_text'])

	models.append(question_model)
	models.append(attention_model)

	if not params['load_roi']:

		# Directly loads the RoI features for all bounding boxes. Hence, no need of instantiating the RoI and Image Emb models.

		image_model = ImageEmbedding(params['feature_type'])
		models.append(image_model)

		if params['use_roi']:
			roi_model = RoIFeats(params['roi_crop'], extra_params['max_num_bars'])
			models.append(roi_model)

	if params['use_text']:

		text_model = TextEmbedding(extra_params['text_vocab_size'], params['emb_size'], extra_params['max_num_text'])
		models.append(text_model)

	print(len(models))
	for m in models:
		print(type(m))
	return models
