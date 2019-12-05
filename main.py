import numpy as np
import sys, time
import cv2
import os
from os.path import join
import argparse
import json
import matplotlib.pyplot as plt
from objectDetection_util import ObjectDetector

def main():
	# add arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', type=str)
	parser.add_argument('--make_data', action='store_true')
	args = parser.parse_args()
	if args.make_data:
		ObjectDetector.makeDataset(args.dir)

	# create object
	detector = ObjectDetector()
	detector.loadData('objects.json', 'data.json')

	while True:
		query = input('Please type your query object: ')
		if query == '!':
			break
		results = detector.retrieveImages(query)
		# print(results)
		detector.outputTargetImages(results, query)


if __name__ == '__main__':
	main()