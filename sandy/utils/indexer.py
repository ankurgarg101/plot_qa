"""
Class definition for the Indexer that holds the bijection mapping between objects and the indices assigned to each object
"""

import json

class Indexer(object):
	def __init__(self, filepath=None):
		self.objs_to_ints = {}
		self.ints_to_objs = {}

		if not (filepath == None):
			self.load(filepath)

	def load(self, filepath):

		with open(filepath, 'r') as idx_file:
			data = json.load(idx_file)

		self.objs_to_ints = data['objs_to_ints']
		self.ints_to_objs = data['ints_to_objs']

	def dump(self, filepath):

		with open(filepath, 'w') as idx_file:
			data = {'objs_to_ints': self.objs_to_ints, 'ints_to_objs': self.ints_to_objs}
			json.dump(data, idx_file, indent=4)

	def __repr__(self):
		return str([str(self.get_object(i)) for i in range(0, len(self))])

	def __len__(self):
		return len(self.objs_to_ints)

	def get_object(self, index):
		if (index not in self.ints_to_objs):
			return None
		else:
			return self.ints_to_objs[index]

	def contains(self, object):
		return self.index_of(object) != -1

	# Returns -1 if the object isn't present, index otherwise
	def index_of(self, object):
		if (object not in self.objs_to_ints):
			return -1
		else:
			return self.objs_to_ints[object]

	# Adds the object to the index if it isn't present, always returns a nonnegative index
	def get_index(self, object, add=True):
		if not add:
			return self.index_of(object)
		if (object not in self.objs_to_ints):
			new_idx = len(self.objs_to_ints)
			self.objs_to_ints[object] = new_idx
			self.ints_to_objs[new_idx] = object
		return self.objs_to_ints[object]