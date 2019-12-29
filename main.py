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

	args = parser.parse_args()
	if args.make_dataset:
		ObjectDetector.makeDataset(args.data_dir)
		exit()

	## create object detector
	detector = ObjectDetector()
	detector.loadData(args.data_dir)

	while True:
		query = input('Please type your query object or type \'!\' to exit: ').strip()
		if query == '!':
			break
		results, labels = detector.retrieveImages(query)
		detector.outputTargetVideos(results)
		# detector.outputTargetImages(results, labels)


if __name__ == '__main__':
	main()