"""
Functions to build the correct models based on the specified parameters.
"""

from .img_emb import ImageEmbedding
from .ques_emb import QuestionEmbedding
from .san import SAN
from .san_multi import SAN_MULTI
from .roi import RoIFeats
from .text_emb import TextEmbedding

def build_models(params, extra_params):

	# Models that will be added irrespective of the other flags

	models = {}

	question_model = QuestionEmbedding(extra_params['ques_vocab_size'], params['emb_size'], params['hidden_size'], params['rnn_size'], params['rnn_layers'], params['dropout'], extra_params['max_ques_seq_len'], params['use_gpu'])

	if params['use_text']:
		if params['n_heads'] > 1:
			attention_model = SAN_MULTI(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'], extra_params['sfeat_text'], img_feat_size=extra_params['img_feat_size'], n_heads = params['n_heads'], dot_product_att = params['dot_product_att'])
		else:
			attention_model = SAN(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'], extra_params['sfeat_text'], img_feat_size=extra_params['img_feat_size'], n_heads = params['n_heads'])
	else:
		if params['n_heads'] > 1:
			attention_model = SAN_MULTI(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'], img_feat_size=extra_params['img_feat_size'],n_heads = params['n_heads'], dot_product_att = params['dot_product_att'])
		else:	
			attention_model = SAN(extra_params['sfeat_img'], extra_params['sfeat_ques'], params['att_size'], extra_params['ans_vocab_size'], params['use_gpu'], img_feat_size=extra_params['img_feat_size'],n_heads = params['n_heads'])

	models['ques_model'] = question_model
	models['att_model'] = attention_model

	if not params['load_roi']:

		# Directly loads the RoI features for all bounding boxes. Hence, no need of instantiating the RoI and Image Emb models.

		image_model = ImageEmbedding(params['feature_type'])
		models['img_model'] = image_model

		if params['use_roi']:
			roi_model = RoIFeats(params['roi_crop'], extra_params['max_num_bars'])
			models['roi_model'] = roi_model

	elif params['use_global_img']:

		image_model = ImageEmbedding(params['feature_type'])
		models['img_model'] = image_model

	if params['use_text']:

		text_model = TextEmbedding(extra_params['text_vocab_size'], params['emb_size'], extra_params['max_num_text'])
		models['text_model'] = text_model

	print(len(models))
	
	return models
