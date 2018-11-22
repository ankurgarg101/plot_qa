import argparse
import copy
import json
import math
from operator import itemgetter
import sys
from pprint import pprint

def _parse_args():
    parser = argparse.ArgumentParser(description = 'preprocess.py')
    
    parser.add_argument('--inp', type=str, required = True, help='path to input metadata file')

    parser.add_argument('--out', type=str, required = True, help='path to output metadata file')

    args = parser.parse_args()
    return args

def bbox_dist(bbox1, bbox2):

    diffx = (bbox1[0]-bbox2[0])**2
    diffy = (bbox1[1]-bbox2[1])**2
    return math.sqrt(diffx + diffy)

def sort_bboxes(bboxes):
    sorted_bboxes = sorted(bboxes, key = itemgetter(0, 1))
    # print (sorted_bboxes)
    final_bboxes = []
    final_bboxes.append(sorted_bboxes[0])
    sorted_bboxes.pop(0)
    num = len(sorted_bboxes)
    for i in range(num):
        min_dist = float('inf')
        min_idx = None
        for j in range(len(sorted_bboxes)):
            d = bbox_dist(final_bboxes[i], sorted_bboxes[j])
            if d < min_dist:
                min_dist = d
                min_idx = j
        assert min_idx is not None
        final_bboxes.append(sorted_bboxes[min_idx])
        sorted_bboxes.pop(min_idx)
    if len(sorted_bboxes) >= 1:
        final_bboxes.append(sorted_bboxes[0])
    return final_bboxes

def test_sort_bboxes():
    bboxes = [(0, 0, 2, 3), (10, 10, 2, 3), (2, 3, 2, 3), (1, 2, 2, 3), (2, 1, 2, 3), (3, 1, 2, 3)]
    final_bboxes = sort_bboxes(bboxes)
    print (final_bboxes)

def read_metadata(input_filename, output_filename):
    f = open(input_filename)
    data = json.load(f)
    print (len(data))
    print (data[0])
    print (data[0].keys())
    for idx in range(len(data)):
        if idx%1000 == 0:
            print ('Example', idx)
        example = data[idx]
        
        texts_bboxes = []
        texts_bboxes_dict = {}
        for text in example['texts']:
            texts_bboxes_dict[tuple(text['bbox'])] = -1
        texts_bboxes = list(texts_bboxes_dict.keys())

        final_bboxes = sort_bboxes(texts_bboxes)
        
        for i in range(len(final_bboxes)):
            k = final_bboxes[i]
            assert k in texts_bboxes_dict
            texts_bboxes_dict[k] = i

        for i in range(len(example['texts'])):
            text = example['texts'][i]
            data[idx]['texts'][i]['idx'] = texts_bboxes_dict[tuple(text['bbox'])]
    f.close()

    g = open(output_filename, 'w')
    json.dump(data, g)
    g.close()

if __name__ == '__main__':
    args = _parse_args()
    print (args)
    # test_sort_bboxes()
    read_metadata(args.input_metadata, args.output_metadata)
