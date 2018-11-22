def concat_bbox(text_feature,bbox):
	#bbox: torch Tensor of size batch x max_num_text x 4
	#text_feature: torch Tensor of size batch x max_num_text x feature_size
	return torch.cat((text_feature,bbox),dim=2)
def concat_class(text_feature,text_class):
	return torch.cat((text_feature,text_class),dim=2)