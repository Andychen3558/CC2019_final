import os
import argparse
from pytube import YouTube

from shotdetect import shotdetect, extract_key_frame
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_dir', default='video', type=str)
	parser.add_argument('--out_dir', default='keyframes', type=str)
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	existed_videos = os.listdir(args.out_dir)

	for url in open('urls.txt','r'):
		url = url.strip()
		name = YouTube(url).streams.first().download()
		shots = shotdetect(name)

		short_name = name[name.rfind('/')+1:name.rfind('.')]
		frame_dir = os.path.join(args.out_dir, short_name)
		
		id = url[url.rfind('=')+1:]
		extract_key_frame(name, shots, frame_dir, id)
		os.system(f'mv \'{name}\' {args.video_dir}')

	# for video_name in os.listdir(args.video_dir):
	# 	name = video_name[:video_name.rfind('.')]
	# 	if name in existed_videos:
	# 		continue

	# 	print('extrace keyframes from', video_name)
	# 	full_video_name = os.path.join(args.video_dir, video_name)
	# 	shots = shotdetect(full_video_name)
	# 	frame_dir = os.path.join(args.out_dir, name)
	# 	extract_key_frame(full_video_name, shots, frame_dir)

