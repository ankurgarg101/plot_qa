"""
Module that trains the SAN network
"""

import argparse
from utils.dataset import PlotDataset


def fetch_args(parser):

	parser.add_argument('--data-dir', type=str, required=True, help="The Path of the dataset containing the images, QA and metadata")
	parser.add_argument('--no-debug', dest='debug', default=True, action='store_false', help="Turn off printing off logs all the modules")
	parser.add_argument('--split', type=str, default='train', help="The dataset split we want to work with for training the model")

	return parser.parse_args()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	args = fetch_args(parser)

	pt = PlotDataset(args)