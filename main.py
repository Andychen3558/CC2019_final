import numpy as np
import sys, time
import cv2
import os
from os.path import join, isdir, isfile
import argparse
import json
import matplotlib.pyplot as plt
from objectDetection_util import ObjectDetector
from shotdetect import shotdetect

def preprocessVideo(video):
	video_id = video.split('.')[0]
	print("video_id is {0}".format(video_id))
	video_id_path = './' + video
	print("video_id_path is {0}".format(video_id_path))
	video_output_dir = './' + video_id
	print("video_output_dir is {0}".format(video_output_dir))
	detect = shotdetect.shotDetector(video_id_path)
	detect.run()
	detect.pick_frame(video_output_dir, video_id)
	return video_output_dir

def main():
	## add arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--make_dataset', action='store_true')
	parser.add_argument('--data_dir', default='keyframes')
	parser.add_argument('--object_file', default='objects.json')
	parser.add_argument('--data_file', default='data.json')
	# parser.add_argument('--video', type=str)
	# args = parser.parse_args()
	# if not isfile(args.video):
	# 	print('[Video doesn\'t exist!]')
	# 	return
	# video_keyframes_dir = './' + args.video.split('.')[0]
	# if not isdir(video_keyframes_dir):
	# 	video_keyframes_dir = preprocessVideo(args.video)
	# 	ObjectDetector.makeDataset(video_keyframes_dir)

	args = parser.parse_args()
	if args.make_dataset:
		ObjectDetector.makeDataset(args.data_dir)
		exit()

	## create object detector
	detector = ObjectDetector()
	# detector.loadData(video_keyframes_dir + '_objects.json', video_keyframes_dir + '_data.json')
	detector.loadData(args.object_file, args.data_file)

	while True:
		query = input('Please type your query object or type \'!\' to exit: ').strip()
		if query == '!':
			break
		results, labels = detector.retrieveImages(query)
		# print(results)
		detector.outputTargetImages(results, labels)


if __name__ == '__main__':
	main()