import os
import argparse
from shotdetect import shotdetect, extract_key_frame
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_dir', default='video', type=str)
	parser.add_argument('--out_dir', default='keyframes', type=str)
	args = parser.parse_args()

	for video_name in os.listdir(args.video_dir):
		video_name = os.path.join(args.video_dir, video_name)
		shots = shotdetect(video_name)
		extract_key_frame(video_name, shots, args.out_dir)