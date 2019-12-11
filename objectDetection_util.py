import numpy as np
import sys, time
import cv2
import os
from os.path import join
import argparse
import json
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from embedding import Embedding

def viewImage(img, file_name):
	# timestamp = file_name.split('@@')[1].rstrip('.jpg') + '(s)'
	cv2.imshow('Display', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


class ObjectDetector():
	def __init__(self):
		self.objects = {}
		self.data = []
		self.embedding = Embedding()

	def loadData(self, object_file, data_file):
		self.objects = json.load(open(object_file, 'r'))
		self.data = json.load(open(data_file, 'r'))

	@staticmethod
	def makeDataset(path):
		print('[Reading dataset and detect objects...]')
		data = []
		objects = {}
		count = 0
		for v in os.listdir(path):
			sub_dir = os.path.join(path, v)
			for f in os.listdir(sub_dir):
				# if count == 100:
				# 	break
				file = join(sub_dir, f)
				print(file)
				img = cv2.imread(file)
				bbox, labels, conf = cv.detect_common_objects(img)
				data.append({'file':file, 'bbox':bbox, 'labels':labels, 'conf':conf})
				## build object table
				for label in labels:
					if label not in objects:
						objects[label] = []
					objects[label].append(len(data)-1)
				count += 1
		for key in objects:
			objects[key] = list(set(objects[key]))
		## write file
		# json.dump(objects, open(path + '_objects.json', 'w'))
		# json.dump(data, open(path + '_data.json', 'w'))

		json.dump(objects, open('objects.json', 'w'))
		json.dump(data, open('data.json', 'w'))

	def retrieveImages(self, query):
		threshold = 0
		# use api to get ranked_list
		ranked_list = self.embedding.retrieve(query)[:10]
		n = len(self.data)
		scores = [0.0] * n

		# ranked_list = [(label, emb_score) for (label, emb_score) in ranked_list if emb_score >= threshold]
		print(ranked_list)
		# compute score using tf-idf
		for (label, emb_score) in ranked_list:
			for idx in self.objects[label]:
				tf = np.log(np.sum([l == label for l in self.data[idx]['labels']]) + 1)
				idf = np.log(n / len(self.objects[label]))
				scores[idx] += tf * idf * emb_score

		# return image indices with top 10 highest scores
		return np.argsort(scores)[::-1][:10], [i[0] for i in ranked_list]

	def outputTargetImages(self, results, query_label):
		if len(results) == 0:
			print('Sorry!')
			return
		for idx in results:
			img = cv2.imread(self.data[idx]['file'])
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			bbox, labels, conf = [], [], []
			for i in range(len(self.data[idx]['labels'])):
				if self.data[idx]['labels'][i] in query_label:
					bbox.append(self.data[idx]['bbox'][i])
					labels.append(self.data[idx]['labels'][i])
					conf.append(self.data[idx]['conf'][i])
			output_image = draw_bbox(img, bbox, labels, conf)
			viewImage(output_image, self.data[idx]['file'])

