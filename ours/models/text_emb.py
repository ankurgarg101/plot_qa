import torch.nn as nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size = None, embedding_size = None, pretrained_matrix = None):
        super(TextEmbedding, self).__init__() # Must call super __init__()
	self.embed = None
	if pretrained_matrix is not None:
		self.embed = nn.Embedding.from_pretrained(pretrained_matrix)
	else:
		self.embed = nn.Embedding(vocab_size,embedding_size)

	def forward(self,index):
		return self.embed(index)