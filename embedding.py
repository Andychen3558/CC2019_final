import gensim.downloader as loader
import json
import numpy as np
class Embedding:
	def __init__(self, vecfile='glove-wiki-gigaword-50'):
		self.model = loader.load(vecfile)
		
		self.labels_emb = []
		self.labels = []


	def count_embedding(self, labels):
		for label in labels:
				self.labels_emb.append(self.w2vec(label))
				self.labels.append(label)
		self.labels_emb = np.array(self.labels_emb)

	def w2vec(self, w):
		if w not in self.model:
			return np.zeros(self.model.vector_size)
		else:
			return self.model[w] / np.linalg.norm(self.model[w])

	def retrieve(self, query, threshold=0):
		qs = query.strip().split()
		total_scroes = np.zeros(len(self.labels))
		total = 0
		for q in qs:
			emb = self.w2vec(q)
			if np.sum(emb) == 0:
				continue
			scores = self.labels_emb @ emb
			total_scroes += scores
			total += 1
		if total != 0:
			total_scroes /= total
		return sorted(list(zip(self.labels, total_scroes)), key=lambda x: x[1], reverse=True)
