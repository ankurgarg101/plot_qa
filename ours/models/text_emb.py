import torch.nn as nn

class TextEmbedding(nn.Module):
	
	def __init__(self, vocab_size, embedding_size, max_num_text, pretrained_matrix = None):
		super(TextEmbedding, self).__init__() # Must call super __init__()
	
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.max_num_text = max_num_text

		if pretrained_matrix is not None:
			self.embed = nn.Embedding.from_pretrained(pretrained_matrix)
		else:
			self.embed = nn.Embedding(vocab_size,embedding_size,padding_idx=0)

	def forward(self, index):
		"""
		index is of shape N, max_num_text
		"""

		index = index
		return self.embed(index)