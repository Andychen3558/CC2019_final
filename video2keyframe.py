import os
from shotdetect.shotdetect import shotDetector as shotDetector
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_dir', default='video', type=str)
	parser.add_argument('--out_dir', default='keyframes', type=str)
	args = parser.parse_args()

	for video_name in os.listdir(args.video_dir):
		label = video_name.split('.')[0]
		print(f'process {video_name}, output to {os.path.join(args.out_dir, label)}')
		detector = shotDetector(os.path.join(args.video_dir, video_name))
		detector.run()
		detector.pick_frame(os.path.join(args.out_dir, label), label)