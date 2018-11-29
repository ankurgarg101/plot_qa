import argparse
import copy
import json
from operator import itemgetter
import sys
from pprint import pprint
import nltk

def _parse_args():
	parser = argparse.ArgumentParser()	
	parser.add_argument('--i', dest='inp_fp', type=str, help='path to input qa data file')
	parser.add_argument('--o', dest='out_fp', type=str, help='path to output metadata file')

	args = parser.parse_args()
	return args

def qa_tokenize(input_filename, output_filename):
	
	print(input_filename)
	with open(input_filename) as f:
		data = json.load(f)
	
	print (len(data))
	print (data[0])
	print (data[0].keys())
	
	for idx in range(len(data)):
		if idx%1000 == 0:
			print ('Example', idx)
		data[idx]['question_tok'] = nltk.word_tokenize(data[idx]['question'])

	f.close()

	g = open(output_filename, 'w')
	json.dump(data, g)
	g.close()

if __name__ == '__main__':
	args = _parse_args()
	print (args)
	print(args.inp_fp, args.out_fp)	
	qa_tokenize(args.inp_fp, args.out_fp)
