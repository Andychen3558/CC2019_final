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

def viewImage(img):
	cv2.imshow('Display', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


class ObjectDetector():
	def __init__(self):
		self.objects = {}
		self.data = []

	def loadData(self, object_file, data_file):
		self.objects = json.load(open(object_file, 'r'))
		self.data = json.load(open(data_file, 'r'))

	@staticmethod
	def makeDataset(path):
		print('[Reading dataset and detect objects...]')
		data = []
		objects = {}
		count = 0
		for f in os.listdir(path):
			if count == 100:
				break
			file = join(path, f)
			img = cv2.imread(file)
			bbox, labels, conf = cv.detect_common_objects(img)
			data.append({'file':file, 'bbox':bbox, 'labels':labels, 'conf':conf})
			# build object table
			for label in labels:
				if label not in objects:
					objects[label] = []
				objects[label].append(len(data)-1)
			count += 1
		for key in objects:
			objects[key] = list(set(objects[key]))
		# write file
		json.dump(objects, open('objects.json', 'w'))
		json.dump(data, open('data.json', 'w'))

	def retrieveImages(self, query):
		if query not in self.objects:
			return None
		return self.objects[query]

	def outputTargetImages(self, results, label):
		if not results:
			print('Sorry!')
			return
		for idx in results:
			img = cv2.imread(self.data[idx]['file'])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			bbox, conf = [], []
			for i in range(len(self.data[idx]['labels'])):
				if self.data[idx]['labels'][i] == label:
					bbox.append(self.data[idx]['bbox'][i])
					conf.append(self.data[idx]['conf'][i])
			output_image = draw_bbox(img, bbox, [label]*len(bbox), conf)
			viewImage(output_image)