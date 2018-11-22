"""
Module that computes the RoI pooling of image features
"""

import torch
from torch import nn
from roi_align.roi_align import RoIAlign

class RoIFeats(nn.Module):

	def __init__(self, crop_dim=7, scale_factor):

		# Check if scale ratio needs to be given as input here or the coordinates should be modified from outside itself.

		# We'll do a square crop of the image features for each bounding box
		self.crop_dim = crop_dim
		self.scale_factor = scale_factor
		self.max_bboxes = max_bboxes

		self.roi_align = RoIAlign(self.crop_dim, self.crop_dim, tranform_fpcoor=False)
		self.avg_pool = nn.AvgPool2d()

	def forward(self, images, boxes, boxes_idx):

		"""
		Make sure that the bounding boxes are normalized wrt to the original image and not the image features
		images: N,C,H,W
		boxes: N, max_bboxes, 4
		boxes_idx: N, 1
		pool_out: N, max_bboxes, C
		"""

		boxes = boxes.view(-1, 4)
		boxes_idx = boxes_idx.view(-1)

		roi_feats = self.roi_align(images, boxes, boxes_idx)

		# Assuming boxes are given in the correct order, just use view. Might need to check

		print("RoI Shape", roi_feats.size())

		pool_out = self.avg_pool(roi_feats)
		print("Pool Out Shape", pool_out.size())
		pool_out = pool_out.view(images.size(0), self.max_bboxes, images.size(1), 1, 1).squeeze()

		return pool_out