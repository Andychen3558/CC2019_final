import numpy as np
import sys, time
import cv2
import os
from os.path import join
import argparse
import json
import matplotlib.pyplot as plt

# from cvlib.object_detection import draw_bbox
from embedding import Embedding
# from moviepy.editor import *

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

	# def loadData(self, object_file, data_file):
	# 	self.objects = json.load(open(object_file, 'r'))
	# 	self.data = json.load(open(data_file, 'r'))

	def loadData(self, data_dir):

		for f in os.listdir(data_dir):
			sub_dir = os.path.join(data_dir, f)
			sub_data = json.load(open(os.path.join(sub_dir, 'data.json'), 'r'))
			sub_object = json.load(open(os.path.join(sub_dir, 'objects.json'), 'r'))
			l = len(self.data)
			for label in sub_object:
				if label not in self.objects:
					self.objects[label] = []
				self.objects[label].extend([i+l for i in sub_object[label]])
			self.data.extend(sub_data)

		self.embedding.count_embedding(self.objects.keys())

	@staticmethod
	def makeDataset(path):
		import cvlib as cv
		print('[Reading dataset and detect objects...]')
		
		for f in os.listdir(path):
			sub_path = join(path, f)
			data = []
			objects = {}
			count = 0
			if 'data.json' in os.listdir(sub_path):
				continue
			for file in os.listdir(sub_path):
				if file.split('.')[-1] == 'json':
					continue
				img_path = join(sub_path, file)
				img = cv2.imread(img_path)
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

			json.dump(objects, open(join(sub_path, 'objects.json'), 'w'))
			json.dump(data, open(join(sub_path, 'data.json'), 'w'))

		## write file
		# json.dump(objects, open(path + '_objects.json', 'w'))
		# json.dump(data, open(path + '_data.json', 'w'))

	def retrieveImages(self, query):
		threshold = 0
		# use api to get ranked_list
		ranked_list = self.embedding.retrieve(query)[:10]
		n = len(self.data)
		scores = [0.0] * n

		# ranked_list = [(label, emb_score) for (label, emb_score) in ranked_list if emb_score >= threshold]
		print('EmbScore:')
		for i in ranked_list:
			print('{}: {:.02f}'.format(i[0], i[1]))
		# compute score using tf-idf
		for (label, emb_score) in ranked_list:
			for idx in self.objects[label]:
				tf = 1
				for i in range(len(self.data[idx]['labels'])):
					if self.data[idx]['labels'][i] == label:
						tf += self.data[idx]['conf'][i]
				tf = np.log(tf)
				# tf = np.log(np.sum([l == label for l in self.data[idx]['labels']]) + 1)
				idf = np.log(n / len(self.objects[label]))
				scores[idx] += tf * idf * emb_score**3

		# return image indices with top 10 highest scores
		return np.argsort(scores)[::-1][:10], [i[0] for i in ranked_list]

	def outputTargetURL(self, query, results):
		if len(results) == 0:
			print('Sorry! There is no results available.')
			return

		urls = []
		for idx in results:
			keyframe = self.data[idx]['file'].rsplit('_', 1)
			video_id = keyframe[0]
			# video_name = 'video/' + keyframe.split('mp4')[0].split('/')[1] + '.mp4'
			timestamp = int(keyframe[1].split('.')[0]) // 1000
			start = timestamp - 1
			url = f'https://youtu.be/{video_id}?t={start}'
			urls.append(url)
		return urls

			# # retrieve video frames
			# clip = VideoFileClip(video_name)
			# print('[video: %s]' %(video_name))
			# print('[timestamp: %f]' %(timestamp))
			# if start < 0:
			# 	start = 0
			# if end > clip.duration:
			# 	end = clip.duration
			# subclip = clip.subclip(start, end)
			# subclip.write_videofile(query + '_' + str(count) + '.mp4')

			# count += 1

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

